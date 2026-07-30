"""
ELO evaluation orchestrator.

Loads single-turn all-layers results, generates pairwise comparisons
at best operating layers, runs LLM judge, and computes ELO ratings.

Stage 1: Per-method coefficient ranking (5 coefs + 2 baselines = 7 players)
Stage 2: Cross-method ranking (best coef per method + 2 baselines = 5 players)

Usage:
    # Stage 1 for a single method:
    python -m src.elo_evaluation.run_elo_evaluation \
        --stage 1 \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait compassion \
        --method swfc \
        --steering_mode all_token_mode

    # Stage 2 (cross-method):
    python -m src.elo_evaluation.run_elo_evaluation \
        --stage 2 \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait compassion \
        --steering_mode all_token_mode

    # Optional flags:
    --honesty_category self_report_experimental  (default for honesty)
    --test  (run on 5 questions only)
    --batch_size 16
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import yaml

from src.constants.constants import (
    RESULTS_FOLDER,
    ROOT_FOLDER,
    get_config_for_trait,
)
from src.elo_evaluation.elo_rating import (
    bootstrap_bt_ci,
    bootstrap_bt_ci_normalized,
    bootstrap_elo_ci,
    compute_bt_ratings,
    compute_elo_ratings,
)
# NOTE: src.elo_evaluation.pairwise_judge pulls in vllm, which is only needed
# when actually running the judge (Stage 1 / Stage 2). It is imported lazily
# inside run_stage1 / run_stage2 so that --recompute_bt and other judge-free
# paths work on machines without a working vllm install.
if TYPE_CHECKING:
    from src.elo_evaluation.pairwise_judge import MatchResult

STEERING_METHODS = ["swfc", "sttp", "stmp"]

TRAIT_DESCRIPTIONS = {
    "compassion": (
        "Compassion is the quality of showing empathy, warmth, and genuine care "
        "for another person's emotional state. A compassionate response validates "
        "feelings, offers support, and creates a sense of connection."
    ),
    "honesty": (
        "Honesty is the quality of truthfully disclosing material facts, even when "
        "the truth is inconvenient or unwelcome. An honest response reveals, explains, "
        "and provides guidance aligned with the truth."
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _model_save_name(model_id: str) -> str:
    return model_id.replace("/", "_")


def _results_dir(trait: str, model_save: str, honesty_category: Optional[str]) -> str:
    base = os.path.join(RESULTS_FOLDER, "logistic_regression", trait, model_save, "all_layers")
    if trait == "honesty" and honesty_category:
        base = os.path.join(base, honesty_category)
    return base


def _load_baseline(
    results_dir: str, trait: str, baseline_type: str, model_save: str, subdir: str = "",
) -> List[dict]:
    """Load a baseline results file (flat list)."""
    filename = f"{trait}_{baseline_type}_baseline_{model_save}_logreg.json"
    path = os.path.join(results_dir, "no_steering", subdir, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in baseline file, got {type(data)}: {path}")
    return data


def _load_steering_results(
    results_dir: str,
    trait: str,
    method: str,
    mode: str,
    model_save: str,
    subdir: str = "",
) -> dict:
    """Load a steering results file (nested dict with layers/coefficients)."""
    filename = f"{trait}_{method}_{mode}_{model_save}_logreg.json"
    path = os.path.join(results_dir, method, subdir, filename)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "layers" not in data:
        raise KeyError(f"No 'layers' key in steering file: {path}")
    return data


def _get_best_layer_config(
    config: dict,
    model_save: str,
    mode: str,
    honesty_category: Optional[str] = None,
) -> dict:
    """
    Extract best-layer config section for a given model, mode.

    Returns dict mapping method -> {layer, coefficient}.
    """
    # Try honesty-category-specific key first, then generic
    if honesty_category:
        key = f"best_layer_{model_save}_{honesty_category}"
        if key in config:
            return config[key][mode]

    key = f"best_layer_{model_save}"
    if key not in config:
        raise KeyError(
            f"No best_layer config for model={model_save!r}. "
            f"Available keys: {[k for k in config if k.startswith('best_layer')]}"
        )
    return config[key][mode]


def _load_ground_truth_honesty(data_path: str) -> Dict[str, dict]:
    """
    Load honesty test scenarios and index by normalized question.

    Returns dict mapping question -> {gold_truth, category}.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", data) if isinstance(data, dict) else data
    gt = {}
    for s in scenarios:
        q = s.get("user_message", "").strip()
        gt[q] = {
            "gold_truth": s.get("gold_truth", ""),
            "category": s.get("category", ""),
        }
    return gt


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _extract_responses_at_layer(
    steering_data: dict,
    layer: int,
    coef_keys: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    """
    Extract responses at a specific layer, keyed by coefficient string.

    Returns dict: coef_key -> list of {question, steered_answer} items.
    """
    layer_str = str(layer)
    layer_data = steering_data["layers"][layer_str]

    if coef_keys is None:
        coef_keys = [k for k in layer_data if k.startswith("coef_")]

    results = {}
    for ck in coef_keys:
        results[ck] = layer_data[ck]
    return results


def _build_player_responses(
    results_dir: str,
    trait: str,
    method: str,
    mode: str,
    model_save: str,
    best_layer: int,
    subdir: str = "",
) -> Dict[str, List[dict]]:
    """
    Build a dict mapping player_name -> list of response items (one per question).

    Players = 5 coefficient variants + 2 baselines.
    """
    # Load steering results at best layer
    steering_data = _load_steering_results(results_dir, trait, method, mode, model_save, subdir)
    coef_responses = _extract_responses_at_layer(steering_data, best_layer)

    # Load baselines
    aligned = _load_baseline(results_dir, trait, "aligned", model_save, subdir)
    malicious = _load_baseline(results_dir, trait, "malicious", model_save, subdir)

    players = {}
    for ck, items in coef_responses.items():
        player_name = f"{method}_{ck}_layer{best_layer}"
        players[player_name] = items

    players["aligned_baseline"] = aligned
    players["malicious_baseline"] = malicious

    return players


def _generate_all_pairs(
    players: Dict[str, List[dict]],
    ground_truth: Optional[Dict[str, dict]] = None,
    n_questions: Optional[int] = None,
) -> List[dict]:
    """
    Generate all pairwise comparison dicts for round-robin tournament.

    Returns list of pair dicts ready for PairwiseJudge.evaluate_pairs_batch().
    """
    player_names = sorted(players.keys())
    n_q = len(next(iter(players.values())))
    if n_questions is not None:
        n_q = min(n_q, n_questions)

    pairs = []
    for p1, p2 in itertools.combinations(player_names, 2):
        for q_idx in range(n_q):
            item1 = players[p1][q_idx]
            item2 = players[p2][q_idx]

            question = item1["question"]
            pair = {
                "question_idx": q_idx,
                "question": question,
                "player_1": p1,
                "player_2": p2,
                "response_1": item1["steered_answer"],
                "response_2": item2["steered_answer"],
            }

            # Add gold_truth for honesty
            if ground_truth is not None:
                gt = ground_truth.get(question.strip(), {})
                pair["gold_truth"] = gt.get("gold_truth", "")

            pairs.append(pair)

    return pairs


# ---------------------------------------------------------------------------
# Saving / loading results
# ---------------------------------------------------------------------------

def _elo_results_dir(trait: str, model_save: str) -> str:
    d = os.path.join(RESULTS_FOLDER, "elo_evaluation", trait, model_save)
    os.makedirs(d, exist_ok=True)
    return d


def _save_results(
    output_dir: str,
    prefix: str,
    matches: List[MatchResult],
    elo_result,
) -> Tuple[str, str]:
    """Save match details and ELO ratings to JSON files."""
    matches_path = os.path.join(output_dir, f"{prefix}_matches.json")
    ratings_path = os.path.join(output_dir, f"{prefix}_ratings.json")

    match_dicts = []
    for m in matches:
        match_dicts.append({
            "question_idx": m.question_idx,
            "player_a": m.player_a,
            "player_b": m.player_b,
            "winner": m.winner,
            "raw_verdict": m.raw_verdict,
            "position_a_was": m.position_a_was,
            "position_b_was": m.position_b_was,
            "judge_raw_output": m.judge_raw_output,
            "parse_failed": m.parse_failed,
            "raw_verdict_forward": m.raw_verdict_forward,
            "raw_verdict_reverse": m.raw_verdict_reverse,
            "judge_raw_output_forward": m.judge_raw_output_forward,
            "judge_raw_output_reverse": m.judge_raw_output_reverse,
        })

    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(match_dicts, f, indent=2, ensure_ascii=False)

    ratings_dict = {
        "ratings": elo_result.ratings,
        "win_matrix": elo_result.win_matrix,
        "tie_matrix": elo_result.tie_matrix,
        "n_matches": elo_result.n_matches,
    }
    if elo_result.bootstrap_ci is not None:
        ratings_dict["bootstrap_ci"] = {
            k: list(v) for k, v in elo_result.bootstrap_ci.items()
        }
    if getattr(elo_result, "bootstrap_ci_normalized", None) is not None:
        ratings_dict["bootstrap_ci_normalized"] = {
            k: list(v) for k, v in elo_result.bootstrap_ci_normalized.items()
        }

    with open(ratings_path, "w", encoding="utf-8") as f:
        json.dump(ratings_dict, f, indent=2, ensure_ascii=False)

    print(f"Saved matches to {matches_path}")
    print(f"Saved ratings to {ratings_path}")
    return matches_path, ratings_path


# ---------------------------------------------------------------------------
# Stage 1: per-method coefficient ranking
# ---------------------------------------------------------------------------

def run_stage1(
    model_id: str,
    trait: str,
    method: str,
    steering_mode: str,
    honesty_category: Optional[str] = None,
    batch_size: Optional[int] = None,
    test: bool = False,
    exp_name: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> None:
    """Run Stage 1 ELO evaluation for a single method."""
    from src.llm_judge.config import JudgeConfig
    from src.elo_evaluation.pairwise_judge import PairwiseJudge

    model_save = _model_save_name(model_id)
    config_path = get_config_for_trait(trait)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    best_layers = _get_best_layer_config(config, model_save, steering_mode, honesty_category)
    method_cfg = best_layers[method]
    best_layer = method_cfg["layer"]

    print(f"Stage 1: {trait} | {model_save} | {method} | {steering_mode} | layer {best_layer}")

    if results_dir is None:
        results_dir = _results_dir(trait, model_save, honesty_category)
    subdir = exp_name or ""
    players = _build_player_responses(
        results_dir, trait, method, steering_mode, model_save, best_layer, subdir,
    )

    print(f"Players ({len(players)}): {sorted(players.keys())}")
    n_q = len(next(iter(players.values())))
    print(f"Questions: {n_q}")

    # Load ground truth for honesty
    ground_truth = None
    if trait == "honesty":
        honesty_cat = honesty_category or "self_report_experimental"
        common_key = f"common_steering_config_{model_save}_{honesty_cat}"
        data_path = os.path.join(
            ROOT_FOLDER, "data",
            config[common_key]["data_path"],
        )
        ground_truth = _load_ground_truth_honesty(data_path)

    n_questions = 5 if test else None
    pairs = _generate_all_pairs(players, ground_truth, n_questions)
    print(f"Total pairs to evaluate: {len(pairs)}")

    # Initialize judge
    judge_config = JudgeConfig.load()
    judge = PairwiseJudge(
        config=judge_config,
        trait=trait,
        trait_description=TRAIT_DESCRIPTIONS[trait],
    )

    match_results = judge.evaluate_pairs_batch(pairs, batch_size=batch_size)

    # Compute ELO
    match_dicts = [
        {"player_a": m.player_a, "player_b": m.player_b, "winner": m.winner}
        for m in match_results
    ]
    elo_result = compute_elo_ratings(match_dicts)

    # Bootstrap CI
    ci = bootstrap_elo_ci(match_dicts)
    elo_result.bootstrap_ci = ci

    # Print results
    print("\n=== Stage 1 ELO Ratings ===")
    for player, rating in elo_result.ratings.items():
        lo, hi = ci.get(player, (0, 0))
        print(f"  {player:50s}  {rating:7.1f}  [{lo:.1f}, {hi:.1f}]")

    # Save
    output_dir = _elo_results_dir(trait, model_save)
    prefix = f"stage1_{method}_{steering_mode}"
    _save_results(output_dir, prefix, match_results, elo_result)


# ---------------------------------------------------------------------------
# Stage 2: cross-method ranking
# ---------------------------------------------------------------------------

def _load_stage1_best_per_method(
    output_dir: str,
    methods: List[str],
    steering_mode: str,
) -> Dict[str, str]:
    """
    Load Stage 1 ratings and find the best coefficient player per method.

    Returns dict: method -> best_player_name.
    """
    best_per_method = {}
    for method in methods:
        ratings_path = os.path.join(output_dir, f"stage1_{method}_{steering_mode}_ratings.json")
        with open(ratings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Find highest-rated non-baseline player
        for player, rating in data["ratings"].items():
            if "baseline" not in player:
                best_per_method[method] = player
                break  # ratings are sorted descending

    return best_per_method


def run_stage2(
    model_id: str,
    trait: str,
    steering_mode: str,
    honesty_category: Optional[str] = None,
    batch_size: Optional[int] = None,
    test: bool = False,
    exp_name: Optional[str] = None,
    results_dir: Optional[str] = None,
) -> None:
    """Run Stage 2 cross-method ELO evaluation."""
    from src.llm_judge.config import JudgeConfig
    from src.elo_evaluation.pairwise_judge import PairwiseJudge

    model_save = _model_save_name(model_id)
    config_path = get_config_for_trait(trait)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if results_dir is None:
        results_dir = _results_dir(trait, model_save, honesty_category)
    subdir = exp_name or ""
    output_dir = _elo_results_dir(trait, model_save)

    # Load Stage 1 winners
    best_per_method = _load_stage1_best_per_method(output_dir, STEERING_METHODS, steering_mode)
    print(f"Stage 2: Best players from Stage 1: {best_per_method}")

    # Build player responses for Stage 2
    best_layers = _get_best_layer_config(config, model_save, steering_mode, honesty_category)
    players: Dict[str, List[dict]] = {}

    for method, player_name in best_per_method.items():
        # Parse layer and coef from player name: e.g., "swfc_coef_3_layer23"
        method_cfg = best_layers[method]
        best_layer = method_cfg["layer"]

        steering_data = _load_steering_results(
            results_dir, trait, method, steering_mode, model_save, subdir,
        )
        layer_data = steering_data["layers"][str(best_layer)]

        # Find the coefficient key from the player name
        # Player name format: "{method}_{coef_key}_layer{N}"
        # e.g., "swfc_coef_3_layer23"
        prefix = f"{method}_"
        suffix = f"_layer{best_layer}"
        coef_key = player_name.removeprefix(prefix).removesuffix(suffix)

        players[player_name] = layer_data[coef_key]

    # Add baselines
    aligned = _load_baseline(results_dir, trait, "aligned", model_save, subdir)
    malicious = _load_baseline(results_dir, trait, "malicious", model_save, subdir)
    players["aligned_baseline"] = aligned
    players["malicious_baseline"] = malicious

    print(f"Stage 2 players ({len(players)}): {sorted(players.keys())}")

    # Load ground truth for honesty
    ground_truth = None
    if trait == "honesty":
        honesty_cat = honesty_category or "self_report_experimental"
        common_key = f"common_steering_config_{model_save}_{honesty_cat}"
        data_path = os.path.join(
            ROOT_FOLDER, "data",
            config[common_key]["data_path"],
        )
        ground_truth = _load_ground_truth_honesty(data_path)

    n_questions = 5 if test else None
    pairs = _generate_all_pairs(players, ground_truth, n_questions)
    print(f"Total pairs to evaluate: {len(pairs)}")

    # Initialize judge
    judge_config = JudgeConfig.load()
    judge = PairwiseJudge(
        config=judge_config,
        trait=trait,
        trait_description=TRAIT_DESCRIPTIONS[trait],
    )

    match_results = judge.evaluate_pairs_batch(pairs, batch_size=batch_size)

    # Compute ELO
    match_dicts = [
        {"player_a": m.player_a, "player_b": m.player_b, "winner": m.winner}
        for m in match_results
    ]
    elo_result = compute_elo_ratings(match_dicts)

    # Bootstrap CI
    ci = bootstrap_elo_ci(match_dicts)
    elo_result.bootstrap_ci = ci

    # Print results
    print("\n=== Stage 2 ELO Ratings (Cross-Method) ===")
    for player, rating in elo_result.ratings.items():
        lo, hi = ci.get(player, (0, 0))
        print(f"  {player:50s}  {rating:7.1f}  [{lo:.1f}, {hi:.1f}]")

    # Save
    prefix = f"stage2_{steering_mode}"
    _save_results(output_dir, prefix, match_results, elo_result)


# ---------------------------------------------------------------------------
# Recompute ratings from existing match files (no judge re-run needed)
# ---------------------------------------------------------------------------

def recompute_bt(
    model_id: str,
    trait: str,
    steering_mode: str,
    stage: int = 1,
) -> None:
    """
    Recompute ratings from saved match files using Bradley-Terry MLE.

    Reads *_matches.json files and writes *_bt_ratings.json files.
    No LLM judge invocation — purely reprocesses existing verdicts.
    """
    model_save = _model_save_name(model_id)
    output_dir = _elo_results_dir(trait, model_save)

    if stage == 1:
        methods = STEERING_METHODS
    else:
        methods = [None]  # stage 2 has no per-method split

    for method in methods:
        if stage == 1:
            prefix = f"stage1_{method}_{steering_mode}"
        else:
            prefix = f"stage2_{steering_mode}"

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

        # Compute BT-MLE ratings
        bt_result = compute_bt_ratings(match_dicts)

        # Bootstrap CIs
        print(f"Computing bootstrap CIs for {prefix} ({len(match_dicts)} matches)...")
        ci = bootstrap_bt_ci(match_dicts)
        bt_result.bootstrap_ci = ci

        ci_norm = bootstrap_bt_ci_normalized(match_dicts)
        bt_result.bootstrap_ci_normalized = ci_norm

        # Print
        method_label = method or "cross-method"
        print(f"\n=== BT-MLE Ratings: {trait} / {method_label} ===")
        for player, rating in bt_result.ratings.items():
            lo, hi = ci.get(player, (0, 0))
            print(f"  {player:50s}  {rating:7.1f}  [{lo:.1f}, {hi:.1f}]")

        # Save alongside existing ELO ratings
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ELO evaluation for steering experiments")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True, choices=["honesty", "compassion"])
    parser.add_argument("--method", type=str, default=None,
                        choices=STEERING_METHODS,
                        help="Required for Stage 1")
    parser.add_argument("--steering_mode", type=str, default="all_token_mode",
                        choices=["all_token_mode", "response_mode"])
    parser.add_argument("--honesty_category", type=str, default="self_report_experimental")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment-name subdir of the steering results")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory holding the technique subdirs "
                             "(swfc/sttp/stmp/no_steering). Overrides the path "
                             "auto-derived from trait/model/honesty_category. "
                             "Combine with --exp_name for the "
                             "run-specific subdir below each technique dir.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Run on 5 questions only")
    parser.add_argument("--recompute_bt", action="store_true",
                        help="Recompute ratings from saved matches using BT-MLE (no judge needed)")
    args = parser.parse_args()

    if args.recompute_bt:
        for trait in ([args.trait] if args.trait else ["honesty", "compassion"]):
            recompute_bt(
                model_id=args.model_id,
                trait=trait,
                steering_mode=args.steering_mode,
                stage=args.stage,
            )
        return

    if args.stage == 1:
        if args.method is None:
            parser.error("--method is required for Stage 1")
        run_stage1(
            model_id=args.model_id,
            trait=args.trait,
            method=args.method,
            steering_mode=args.steering_mode,
            honesty_category=args.honesty_category if args.trait == "honesty" else None,
            batch_size=args.batch_size,
            test=args.test,
            exp_name=args.exp_name,
            results_dir=args.results_dir,
        )
    elif args.stage == 2:
        run_stage2(
            model_id=args.model_id,
            trait=args.trait,
            steering_mode=args.steering_mode,
            honesty_category=args.honesty_category if args.trait == "honesty" else None,
            batch_size=args.batch_size,
            test=args.test,
            exp_name=args.exp_name,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
