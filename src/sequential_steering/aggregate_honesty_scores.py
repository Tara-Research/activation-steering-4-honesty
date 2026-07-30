"""
Aggregate honesty (and coherence) judge scores inside a steering result file
and write the aggregates back under a top-level ``scores`` block.

Input file shape (already produced by the LLM judge):

    {
      "common_steering_config": {...},
      "specific_config": {...},
      "layers": {
        "<layer_idx>": {
          "coef_<N>": [
            {
              "scenario_id": ...,
              "category": ...,
              "question": ...,
              "steered_answer": ...,
              "judge_score": {"honesty_score": <int 0-100>, ...},
              "coherence_score": <int 0-100>,
              ...
            }, ...
          ]
        }
      }
    }

After running this script, the same file will carry an additional block::

    "scores": {
        "<layer_idx>": {
            "coef_<N>": {
                "overall": {
                    "honesty_mean": 73.4, "honesty_n": 50,
                    "coherence_mean": 88.1, "coherence_n": 50,
                    "parse_errors": 0
                },
                "by_category": {
                    "institutional_disclosure": {
                        "honesty_mean": 68.0, "honesty_n": 10,
                        "coherence_mean": 85.5, "coherence_n": 10
                    },
                    ...
                }
            }
        }
    }

Usage:
    python -m src.sequential_steering.aggregate_honesty_scores \
        --file results/.../honesty_sttp_all_token_mode_..._logreg.json

    # for older result files without `category` on each row:
    python -m src.sequential_steering.aggregate_honesty_scores \
        --file <path> --scenarios data/honesty/CAA/generalized/generalized_honesty_scenarios.json
"""

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from src.llm_judge.runners.ground_truth import load_ground_truth_data, normalize_message



def _extract_honesty(row: Dict[str, Any]) -> Optional[int]:
    js = row.get("judge_score")
    if not isinstance(js, dict):
        return None
    if js.get("parse_error"):
        return None
    score = js.get("honesty_score", js.get("score"))
    return int(score) if isinstance(score, (int, float)) else None


def _extract_coherence(row: Dict[str, Any]) -> Optional[int]:
    score = row.get("coherence_score")
    return int(score) if isinstance(score, (int, float)) else None


def _resolve_category(row: Dict[str, Any], msg_to_scenario: Optional[Dict[str, Dict[str, Any]]]) -> Optional[str]:
    cat = row.get("category")
    if cat:
        return cat
    if msg_to_scenario is None:
        return None
    q = row.get("question", "")
    hit = msg_to_scenario.get(normalize_message(q))
    return hit.get("category") if hit else None


def _aggregate_rows(
    rows: List[Dict[str, Any]],
    msg_to_scenario: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    honesty_all: List[int] = []
    coherence_all: List[int] = []
    parse_errors = 0

    by_cat_honesty: Dict[str, List[int]] = defaultdict(list)
    by_cat_coherence: Dict[str, List[int]] = defaultdict(list)

    for row in rows:
        h = _extract_honesty(row)
        c = _extract_coherence(row)
        if h is None:
            parse_errors += 1
        else:
            honesty_all.append(h)
        if c is not None:
            coherence_all.append(c)

        cat = _resolve_category(row, msg_to_scenario) or "_unknown"
        if h is not None:
            by_cat_honesty[cat].append(h)
        if c is not None:
            by_cat_coherence[cat].append(c)

    def _summary(honesty: List[int], coherence: List[int]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if honesty:
            out["honesty_mean"] = round(mean(honesty), 2)
            out["honesty_n"] = len(honesty)
        if coherence:
            out["coherence_mean"] = round(mean(coherence), 2)
            out["coherence_n"] = len(coherence)
        return out

    overall = _summary(honesty_all, coherence_all)
    overall["parse_errors"] = parse_errors

    by_category: Dict[str, Dict[str, Any]] = {}
    all_cats = set(by_cat_honesty) | set(by_cat_coherence)
    for cat in sorted(all_cats):
        by_category[cat] = _summary(by_cat_honesty.get(cat, []), by_cat_coherence.get(cat, []))

    return {"overall": overall, "by_category": by_category}


def aggregate_file(file_path: str, scenarios_path: Optional[str] = None) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "layers" not in data:
        raise ValueError(f"{file_path} has no 'layers' block — not a steering result file")

    msg_to_scenario = None
    if scenarios_path:
        msg_to_scenario = load_ground_truth_data(scenarios_path)

    # Peek at one row to warn if category is missing and no scenarios file was given.
    if msg_to_scenario is None:
        for layer_block in data["layers"].values():
            if isinstance(layer_block, list):
                sample_rows: Iterable[Dict[str, Any]] = layer_block
            elif isinstance(layer_block, dict):
                sample_rows = next((v for v in layer_block.values() if isinstance(v, list)), [])
            else:
                sample_rows = []
            for row in sample_rows:
                if not row.get("category"):
                    print("[aggregate] WARNING: rows have no 'category' field and "
                          "no --scenarios was provided. Per-category scores will be "
                          "collapsed under '_unknown'.")
                break
            break

    scores: Dict[str, Any] = {}
    for layer_key, layer_block in data["layers"].items():
        if isinstance(layer_block, list):
            # baseline format: layer -> list of rows (no coef split)
            scores[layer_key] = {"_default": _aggregate_rows(layer_block, msg_to_scenario)}
            continue
        layer_scores: Dict[str, Any] = {}
        for coef_key, rows in layer_block.items():
            if not isinstance(rows, list):
                continue
            layer_scores[coef_key] = _aggregate_rows(rows, msg_to_scenario)
        scores[layer_key] = layer_scores

    data["scores"] = scores

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return scores


def _print_summary(file_path: str, scores: Dict[str, Any]) -> None:
    print(f"\n[aggregate] wrote scores -> {file_path}")
    for layer_key in sorted(scores.keys(), key=lambda k: (len(k), k)):
        layer_scores = scores[layer_key]
        for coef_key, block in layer_scores.items():
            overall = block.get("overall", {})
            h = overall.get("honesty_mean")
            c = overall.get("coherence_mean")
            n = overall.get("honesty_n", overall.get("coherence_n", 0))
            errs = overall.get("parse_errors", 0)
            extra = f", parse_errors={errs}" if errs else ""
            print(f"  layer {layer_key} {coef_key}: "
                  f"honesty={h} coherence={c} (n={n}{extra})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate honesty/coherence scores into a steering result file.")
    parser.add_argument("--file", required=True, help="Path to the result JSON.")
    parser.add_argument("--scenarios", default=None,
                        help="Optional scenarios JSON for category resolution on rows missing 'category'.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-layer summary output.")
    args = parser.parse_args()

    scores = aggregate_file(args.file, scenarios_path=args.scenarios)
    if not args.quiet:
        _print_summary(args.file, scores)


if __name__ == "__main__":
    main()
