"""Stage 3: Aggregate MT-Bench judge scores into summary metrics.

Usage:
    python -m src.mt_bench.metrics \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait honesty \
        --techniques no_steering swfc
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from src.constants.constants import MT_BENCH_RESULTS_DIR, VECTOR_TYPE
from src.benchmarking.mt_bench.data_utils import MT_BENCH_CATEGORIES


def _load_judgments(judgment_file: str) -> list:
    """Load all judgment records from JSONL."""
    records = []
    with open(judgment_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _compute_metrics_for_records(records: list) -> dict:
    """Compute aggregate metrics from a list of judgment records.

    Returns:
        Dict with overall_mean, overall_std, turn_1_mean, turn_2_mean,
        n_scored, n_failed, and by_category breakdown.
    """
    all_scores = []
    turn_1_scores = []
    turn_2_scores = []
    n_failed = 0

    by_category = defaultdict(lambda: {"turn_1": [], "turn_2": [], "all": []})

    for rec in records:
        score = rec["score"]
        turn = rec["turn"]
        category = rec["category"]

        if score < 0:
            n_failed += 1
            continue

        all_scores.append(score)
        if turn == 1:
            turn_1_scores.append(score)
            by_category[category]["turn_1"].append(score)
        else:
            turn_2_scores.append(score)
            by_category[category]["turn_2"].append(score)
        by_category[category]["all"].append(score)

    turn_1_n = len(turn_1_scores)
    turn_2_n = len(turn_2_scores)
    turn_1_std = float(np.std(turn_1_scores)) if turn_1_scores else 0.0
    turn_2_std = float(np.std(turn_2_scores)) if turn_2_scores else 0.0

    result = {
        "overall_mean": float(np.mean(all_scores)) if all_scores else 0.0,
        "overall_std": float(np.std(all_scores)) if all_scores else 0.0,
        "turn_1_mean": float(np.mean(turn_1_scores)) if turn_1_scores else 0.0,
        "turn_1_std": turn_1_std,
        "turn_1_se": turn_1_std / np.sqrt(turn_1_n) if turn_1_n > 0 else 0.0,
        "turn_1_n": turn_1_n,
        "turn_2_mean": float(np.mean(turn_2_scores)) if turn_2_scores else 0.0,
        "turn_2_std": turn_2_std,
        "turn_2_se": turn_2_std / np.sqrt(turn_2_n) if turn_2_n > 0 else 0.0,
        "turn_2_n": turn_2_n,
        "n_scored": len(all_scores),
        "n_failed": n_failed,
        "by_category": {},
    }

    for category in MT_BENCH_CATEGORIES:
        cat_data = by_category[category]
        if cat_data["all"]:
            result["by_category"][category] = {
                "mean": float(np.mean(cat_data["all"])),
                "std": float(np.std(cat_data["all"])),
                "turn_1_mean": float(np.mean(cat_data["turn_1"])) if cat_data["turn_1"] else 0.0,
                "turn_2_mean": float(np.mean(cat_data["turn_2"])) if cat_data["turn_2"] else 0.0,
                "n": len(cat_data["all"]),
            }

    return result


def _normalize_coef_key(coef) -> str:
    """Normalize coefficient to a string key: float values become str(float), None becomes 'null'."""
    if coef is None:
        return "null"
    return str(float(coef))


def compute_metrics(judgment_file: str) -> dict:
    """Compute per-coefficient metrics from a judgment JSONL file.

    Returns:
        Dict with "coefficients" key mapping coef_str -> metrics dict.
    """
    records = _load_judgments(judgment_file)

    # Group records by coefficient
    coef_groups = defaultdict(list)
    for rec in records:
        coef_key = _normalize_coef_key(rec.get("coefficient"))
        coef_groups[coef_key].append(rec)

    coefficients = {}
    for coef_key, group_records in sorted(coef_groups.items(), key=lambda x: (x[0] != "null", x[0])):
        coefficients[coef_key] = _compute_metrics_for_records(group_records)

    return {"coefficients": coefficients}


def _print_summary(techniques_metrics: dict, model_id: str):
    """Print a formatted summary table to stdout."""
    print(f"\n{'='*80}")
    print(f"MT-Bench Results: {model_id}")
    print(f"{'='*80}")

    header = f"{'Technique':<25} {'Coeff':>8} {'Overall':>8} {'Turn 1':>8} {'Turn 2':>8} {'N':>5} {'Fail':>5}"
    print(header)
    print("-" * 80)

    for technique, data in techniques_metrics.items():
        for coef_key, metrics in data["coefficients"].items():
            coef_display = coef_key if coef_key != "null" else "—"
            print(
                f"{technique:<25} "
                f"{coef_display:>8} "
                f"{metrics['overall_mean']:>7.2f}  "
                f"{metrics['turn_1_mean']:>7.2f}  "
                f"{metrics['turn_2_mean']:>7.2f}  "
                f"{metrics['n_scored']:>5}  "
                f"{metrics['n_failed']:>5}"
            )

    print(f"\n{'By Category':}")
    # Collect all (technique, coef) pairs for column headers
    columns = []
    for technique, data in techniques_metrics.items():
        for coef_key in data["coefficients"]:
            columns.append((technique, coef_key))

    col_width = 12
    print(f"{'Category':<15}", end="")
    for technique, coef_key in columns:
        label = f"{technique[:8]}|{coef_key[:5]}"
        print(f" {label:>{col_width}}", end="")
    print()
    print("-" * (15 + (col_width + 1) * len(columns)))

    for category in MT_BENCH_CATEGORIES:
        print(f"{category:<15}", end="")
        for technique, coef_key in columns:
            cat = techniques_metrics[technique]["coefficients"][coef_key]["by_category"].get(category)
            if cat:
                print(f" {cat['mean']:>{col_width}.2f}", end="")
            else:
                print(f" {'N/A':>{col_width}}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute MT-Bench aggregate metrics from judge scores"
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--trait", required=True, help="Trait name")
    parser.add_argument(
        "--techniques", nargs="+", required=True,
        help="Techniques to aggregate",
    )
    parser.add_argument("--exp_name", required=True, help="Experiment name (used in result directory)")

    args = parser.parse_args()

    model_save_name = args.model_id.replace("/", "_")
    results_dir = os.path.join(
        MT_BENCH_RESULTS_DIR, VECTOR_TYPE, args.trait, model_save_name,
        args.exp_name,
    )

    techniques_metrics = {}
    for technique in args.techniques:
        judgment_file = os.path.join(results_dir, f"model_judgment_{technique}.jsonl")
        if not os.path.exists(judgment_file):
            print(f"Skipping {technique}: judgment file not found: {judgment_file}")
            continue

        techniques_metrics[technique] = compute_metrics(judgment_file)

    if not techniques_metrics:
        print("No judgment files found. Nothing to aggregate.")
        return

    # Save metrics JSON
    output = {
        "model": model_save_name,
        "vector_type": VECTOR_TYPE,
        "trait": args.trait,
        "techniques": techniques_metrics,
    }
    metrics_file = os.path.join(results_dir, "mt_bench_metrics.json")
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    _print_summary(techniques_metrics, args.model_id)


if __name__ == "__main__":
    main()
