"""
Backfill ``scenario_id`` and ``category`` onto every response row in existing
steering result files by matching ``row["question"]`` against
``scenario["user_message"]`` in a reference scenarios JSON.

Walks a directory recursively and updates every file that looks like a
steering result (has a ``layers`` block). Idempotent: rows that already have
both fields are skipped unless ``--force`` is passed. Files with no matches
are left untouched.

Usage:
    python -m src.sequential_steering.backfill_scenario_meta \
        --scenarios data/honesty/CAA/generalized/generalized_honesty_scenarios.json \
        --dir results/logistic_regression/test/honesty

    # dry-run first to see what would change:
    python -m src.sequential_steering.backfill_scenario_meta \
        --scenarios <path> --dir <path> --dry_run
"""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from src.llm_judge.runners.ground_truth import normalize_message


def load_scenario_index(scenarios_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a map from normalized user_message -> {"scenario_id": ..., "category": ...}.

    Supports both the dict-with-"scenarios" shape (generalized/diverse/prd) and
    the flat-list shape (fever). For fever there is no scenario_id/category,
    so we fabricate scenario_id=str(id), category="fever".
    """
    with open(scenarios_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios: Iterable[Dict[str, Any]]
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
        is_fever = False
    elif isinstance(data, list):
        scenarios = data
        is_fever = True
    else:
        raise ValueError(f"{scenarios_path} has unexpected shape (not a list and no 'scenarios' key)")

    index: Dict[str, Dict[str, Any]] = {}
    for scen in scenarios:
        msg = scen.get("user_message")
        if not msg:
            continue
        key = normalize_message(msg)
        if is_fever:
            entry = {"scenario_id": str(scen.get("id")), "category": "fever"}
        else:
            entry = {
                "scenario_id": scen.get("scenario_id"),
                "category": scen.get("category"),
            }
        index[key] = entry
    return index


def _iter_row_lists(data: Dict[str, Any]) -> Iterable[List[Dict[str, Any]]]:
    """Yield every list of response rows inside ``data["layers"]``."""
    layers = data.get("layers")
    if not isinstance(layers, dict):
        return
    for layer_block in layers.values():
        if isinstance(layer_block, list):
            yield layer_block
        elif isinstance(layer_block, dict):
            for v in layer_block.values():
                if isinstance(v, list):
                    yield v


def backfill_file(
    file_path: str,
    index: Dict[str, Dict[str, Any]],
    *,
    force: bool = False,
) -> Tuple[int, int, int]:
    """
    Returns (updated, skipped_already_set, unmatched).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "layers" not in data:
        return (0, 0, 0)

    updated = 0
    already_set = 0
    unmatched = 0

    for rows in _iter_row_lists(data):
        for row in rows:
            if not isinstance(row, dict):
                continue

            has_both = row.get("scenario_id") and row.get("category")
            if has_both and not force:
                already_set += 1
                continue

            q = row.get("question", "")
            hit = index.get(normalize_message(q))
            if not hit:
                unmatched += 1
                continue

            if force or not row.get("scenario_id"):
                row["scenario_id"] = hit["scenario_id"]
            if force or not row.get("category"):
                row["category"] = hit["category"]
            updated += 1

    if updated > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return (updated, already_set, unmatched)


def walk_result_files(root: str) -> Iterable[str]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            # Skip the separate metrics sidecar files.
            if name.endswith("_metrics.json") or name.endswith("_n_run_metrics.json"):
                continue
            yield os.path.join(dirpath, name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill scenario_id + category onto response rows in steering result files."
    )
    parser.add_argument("--scenarios", required=True, help="Reference scenarios JSON.")
    parser.add_argument("--dir", required=True, help="Directory to walk recursively for result files.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing scenario_id/category on rows.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would change without writing any files.")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"--dir is not a directory: {args.dir}")

    index = load_scenario_index(args.scenarios)
    print(f"[backfill] indexed {len(index)} scenarios from {args.scenarios}")

    total_files = 0
    touched_files = 0
    total_updated = 0
    total_unmatched = 0
    total_skipped = 0

    for path in walk_result_files(args.dir):
        total_files += 1

        if args.dry_run:
            # Count what would change without mutating the file.
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            if not isinstance(data, dict) or "layers" not in data:
                continue
            u = s = un = 0
            for rows in _iter_row_lists(data):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if row.get("scenario_id") and row.get("category") and not args.force:
                        s += 1
                        continue
                    hit = index.get(normalize_message(row.get("question", "")))
                    if hit:
                        u += 1
                    else:
                        un += 1
        else:
            u, s, un = backfill_file(path, index, force=args.force)

        if u > 0:
            touched_files += 1
            print(f"  {'[dry]' if args.dry_run else '[ok]'} {path}: "
                  f"updated={u} already_set={s} unmatched={un}")
        total_updated += u
        total_skipped += s
        total_unmatched += un

    verb = "would update" if args.dry_run else "updated"
    print(f"\n[backfill] scanned {total_files} file(s); {verb} {total_updated} row(s) "
          f"across {touched_files} file(s). "
          f"already_set={total_skipped}, unmatched={total_unmatched}.")


if __name__ == "__main__":
    main()
