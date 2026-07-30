"""
Pick the best layer for each steering vector in a layer-sweep results directory.

Given a directory containing one or more layer-sweep result files (one per
steering vector), this script:

1. Aggregates per-layer mean honesty + mean coherence over all rows.
2. Filters layers where ``coherence_mean >= --coherence-threshold``.
3. Picks the layer with the highest ``honesty_mean`` per vector.

The expected directory layout matches what the layer-sweep experiments emit
under e.g.::

    results/.../sttp/
        honesty_sttp_all_token_mode_<model>_logreg.json                # generalized vector
        per_category_<cat>/honesty_sttp_all_token_mode_<model>_logreg.json
        ...

Output: ``best_layer_per_vector.json`` written into ``--root``, plus a summary
table printed to stdout.

Usage::

    python -m src.sequential_steering.find_best_layer_per_vector \
        --root results/logistic_regression/test/layer_sweep/l20-40-s1/honesty/test_meta-llama_Llama-3.3-70B-Instruct/all_layers/generalized/sttp \
        --coherence-threshold 80
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.sequential_steering.aggregate_honesty_scores import _aggregate_rows


def _vector_label(file_path: Path, root: Path) -> str:
    parent = file_path.parent
    if parent == root:
        return "generalized"
    return parent.name


def _coef_int(coef_key: str) -> Optional[int]:
    if coef_key.startswith("coef_"):
        try:
            return int(coef_key.split("_", 1)[1])
        except ValueError:
            return None
    return None


def _per_vector_summary(
    file_path: Path,
    coherence_threshold: float,
) -> Dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "layers" not in data:
        raise ValueError(f"{file_path} has no 'layers' block")

    common_cfg = data.get("common_steering_config", {})
    vector_path = common_cfg.get("vector_path")

    per_layer: List[Dict[str, Any]] = []
    for layer_key, layer_block in data["layers"].items():
        try:
            layer_idx = int(layer_key)
        except ValueError:
            continue
        if not isinstance(layer_block, dict):
            continue
        for coef_key, rows in layer_block.items():
            if not isinstance(rows, list):
                continue
            agg = _aggregate_rows(rows, msg_to_scenario=None)
            overall = agg["overall"]
            entry: Dict[str, Any] = {
                "layer": layer_idx,
                "coef": _coef_int(coef_key),
                "honesty_mean": overall.get("honesty_mean"),
                "honesty_n": overall.get("honesty_n", 0),
                "coherence_mean": overall.get("coherence_mean"),
                "coherence_n": overall.get("coherence_n", 0),
                "parse_errors": overall.get("parse_errors", 0),
            }
            per_layer.append(entry)

    per_layer.sort(key=lambda e: (e["layer"], e["coef"] if e["coef"] is not None else -1))

    eligible = [
        e for e in per_layer
        if e["honesty_mean"] is not None
        and e["coherence_mean"] is not None
        and e["coherence_mean"] >= coherence_threshold
    ]

    best: Optional[Dict[str, Any]] = None
    if eligible:
        winner = max(eligible, key=lambda e: e["honesty_mean"])
        best = {
            "layer": winner["layer"],
            "coef": winner["coef"],
            "honesty_mean": winner["honesty_mean"],
            "coherence_mean": winner["coherence_mean"],
            "n": winner["honesty_n"],
        }

    return {
        "vector_path": vector_path,
        "best": best,
        "per_layer": per_layer,
    }


def find_best_layers(root: Path, coherence_threshold: float) -> Dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"--root does not exist or is not a directory: {root}")

    files = sorted(root.glob("**/honesty_*_logreg.json"))
    if not files:
        raise FileNotFoundError(f"No 'honesty_*_logreg.json' files under {root}")

    vectors: Dict[str, Any] = {}
    method: Optional[str] = None
    steering_mode: Optional[str] = None
    model: Optional[str] = None

    for fp in files:
        label = _vector_label(fp, root)
        if label in vectors:
            raise ValueError(f"Duplicate vector label '{label}' from {fp}")
        summary = _per_vector_summary(fp, coherence_threshold)
        vectors[label] = summary

        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        common_cfg = data.get("common_steering_config", {})
        method = method or common_cfg.get("steering_technique")
        token_mode = common_cfg.get("steering_type")
        if token_mode and steering_mode is None:
            steering_mode = "all_token_mode" if token_mode == "all" else "response_mode"
        model = model or common_cfg.get("model_id") or common_cfg.get("model_name")

        if summary["best"] is None:
            print(
                f"[find_best_layer] WARNING: no layer for vector '{label}' clears "
                f"coherence>={coherence_threshold}",
                file=sys.stderr,
            )

    ordered_labels = (
        ["generalized"] if "generalized" in vectors else []
    ) + sorted(k for k in vectors if k != "generalized")

    summary_table = []
    for label in ordered_labels:
        b = vectors[label]["best"]
        summary_table.append(
            {
                "vector": label,
                "best_layer": b["layer"] if b else None,
                "coef": b["coef"] if b else None,
                "honesty_mean": b["honesty_mean"] if b else None,
                "coherence_mean": b["coherence_mean"] if b else None,
            }
        )

    return {
        "root": str(root).replace("\\", "/"),
        "method": method,
        "steering_mode": steering_mode,
        "model": model,
        "coherence_threshold": coherence_threshold,
        "vectors": {label: vectors[label] for label in ordered_labels},
        "summary_table": summary_table,
    }


def _print_table(summary: Dict[str, Any]) -> None:
    rows = summary["summary_table"]
    header = f"{'vector':<42} {'layer':>5} {'coef':>5} {'honesty':>8} {'coherence':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        layer = row["best_layer"] if row["best_layer"] is not None else "-"
        coef = row["coef"] if row["coef"] is not None else "-"
        honesty = f"{row['honesty_mean']:.2f}" if row["honesty_mean"] is not None else "-"
        coherence = f"{row['coherence_mean']:.2f}" if row["coherence_mean"] is not None else "-"
        print(f"{row['vector']:<42} {str(layer):>5} {str(coef):>5} {honesty:>8} {coherence:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Directory containing per-vector layer-sweep result files.")
    parser.add_argument("--coherence-threshold", type=float, default=80.0,
                        help="Minimum coherence_mean a layer must reach to be eligible (default: 80).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path. Defaults to <root>/best_layer_per_vector.json.")
    args = parser.parse_args()

    summary = find_best_layers(args.root, args.coherence_threshold)
    out_path = args.output or (args.root / "best_layer_per_vector.json")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _print_table(summary)
    print(f"\n[find_best_layer] wrote -> {out_path}")


if __name__ == "__main__":
    main()
