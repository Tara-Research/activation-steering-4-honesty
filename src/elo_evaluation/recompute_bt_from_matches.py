"""
Recompute ratings from saved match files using Bradley-Terry MLE.

No GPU or vLLM needed — purely reprocesses existing judge verdicts.

Usage:
    # Recompute all Stage 1 tournaments for a model
    python -m src.elo_evaluation.recompute_bt_from_matches \
        --model_id meta-llama/Llama-3.3-70B-Instruct

    # Specific trait and stage
    python -m src.elo_evaluation.recompute_bt_from_matches \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait honesty --stage 2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from src.constants.constants import RESULTS_FOLDER
from src.elo_evaluation.elo_rating import (
    bootstrap_bt_ci,
    bootstrap_bt_ci_normalized,
    compute_bt_ratings,
)


STEERING_METHODS = ["swfc", "sttp", "stmp"]


def _elo_results_dir(trait: str, model_save: str) -> str:
    return os.path.join(RESULTS_FOLDER, "elo_evaluation", trait, model_save)


def recompute(
    model_id: str,
    trait: str,
    steering_mode: str,
    stage: int,
    n_bootstrap: int = 1000,
) -> None:
    """Recompute ratings from saved match files using BT-MLE."""
    model_save = model_id.replace("/", "_")
    output_dir = _elo_results_dir(trait, model_save)

    if stage == 1:
        prefixes = [f"stage1_{m}_{steering_mode}" for m in STEERING_METHODS]
    else:
        prefixes = [f"stage2_{steering_mode}"]

    for prefix in prefixes:
        matches_path = os.path.join(output_dir, f"{prefix}_matches.json")
        if not os.path.exists(matches_path):
            print(f"Skipping: {matches_path} not found")
            continue

        with open(matches_path, "r", encoding="utf-8") as f:
            matches = json.load(f)

        match_dicts = [
            {"player_a": m["player_a"], "player_b": m["player_b"], "winner": m["winner"]}
            for m in matches
        ]

        print(f"\n{'='*70}")
        print(f"BT-MLE: {trait} / {prefix} ({len(match_dicts)} matches)")
        print(f"{'='*70}")

        bt_result = compute_bt_ratings(match_dicts)

        print(f"Computing {n_bootstrap} bootstrap resamples...")
        ci = bootstrap_bt_ci(match_dicts, n_bootstrap=n_bootstrap)
        bt_result.bootstrap_ci = ci

        ci_norm = bootstrap_bt_ci_normalized(match_dicts, n_bootstrap=n_bootstrap)
        bt_result.bootstrap_ci_normalized = ci_norm

        for player, rating in bt_result.ratings.items():
            lo, hi = ci.get(player, (0, 0))
            print(f"  {player:50s}  {rating:7.1f}  [{lo:.1f}, {hi:.1f}]")

        bt_ratings_path = os.path.join(output_dir, f"{prefix}_bt_ratings.json")
        ratings_dict = {
            "ratings": bt_result.ratings,
            "win_matrix": bt_result.win_matrix,
            "tie_matrix": bt_result.tie_matrix,
            "n_matches": bt_result.n_matches,
            "method": "bradley_terry_mle",
        }
        if bt_result.bootstrap_ci is not None:
            ratings_dict["bootstrap_ci"] = {
                k: list(v) for k, v in bt_result.bootstrap_ci.items()
            }
        if bt_result.bootstrap_ci_normalized is not None:
            ratings_dict["bootstrap_ci_normalized"] = {
                k: list(v) for k, v in bt_result.bootstrap_ci_normalized.items()
            }
        with open(bt_ratings_path, "w", encoding="utf-8") as f:
            json.dump(ratings_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved: {bt_ratings_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute ratings from saved matches using Bradley-Terry MLE"
    )
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--trait", type=str, default=None,
                        choices=["honesty", "compassion"],
                        help="Trait (default: both)")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--steering_mode", type=str, default="all_token_mode")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    traits = [args.trait] if args.trait else ["honesty", "compassion"]
    for trait in traits:
        recompute(
            model_id=args.model_id,
            trait=trait,
            steering_mode=args.steering_mode,
            stage=args.stage,
            n_bootstrap=args.n_bootstrap,
        )


if __name__ == "__main__":
    main()
