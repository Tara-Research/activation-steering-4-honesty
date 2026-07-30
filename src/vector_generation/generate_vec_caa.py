"""
CAA (Contrastive Activation Addition) vector extraction, unified across traits.

Reads the responses JSON produced by generate_responses.py (after the optional
LLM-judge annotation pass), builds matched pos/neg pairs via a per-trait
adapter, and extracts the response-average activation per layer:

  steering vector [L, D] = mean over pairs of (pos_response_avg - neg_response_avg)

Also saves the per-side response-average embeddings [L, N, D] (the input to the
logistic-regression fit) and, when --per-token-layer is given, ragged per-token
response embeddings for that layer (used by the PCA / histogram analyses).

Usage:
    python -m src.vector_generation.generate_vec_caa --trait honesty \
        --dataset self_report_experimental --model-id meta-llama/Llama-3.3-70B-Instruct
    python -m src.vector_generation.generate_vec_caa --trait compassion \
        --dataset compassion --model-id meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.constants import constants
from src.constants.constants import DATA_FOLDER, PERSONA_VECTORS_FOLDER
from src.utils.model_utils import load_model, set_seed
from src.vector_generation.extraction_utils import (
    compute_paired_mean_diff_vector,
    ensure_padding,
    extract_response_embeddings_in_order,
    load_compassion_system_prompts,
    model_slug,
    save_per_token_embeddings,
    save_side_embeddings,
    save_steering_vector,
)

SIDE_NAMES = {
    "honesty": ("honest", "dishonest"),
    "compassion": ("positive", "negative"),
}


# =====================================================================
# TRAIT ADAPTERS — both emit pairs of the form
#   {"pair_id": str,
#    "pos": {"pair_id", "system_prompt", "user_prompt", "response"},
#    "neg": {"pair_id", "system_prompt", "user_prompt", "response"}}
# =====================================================================

def load_honesty_pairs(
    json_path: Union[str, Path],
    filter_by_judge: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load a scenario-based honesty responses JSON and build matched pairs.

    Each scenario has structure:
      {
        "scenario_id": "...",
        "user_message": "...",
        "responses": {
          "honest":    {"system_prompt": "...", "generations": [{"response": "...", "judge_caa_*": ...}]},
          "dishonest": {"system_prompt": "...", "generations": [{"response": "...", "judge_caa_*": ...}]},
        },
      }

    When filter_by_judge is True and judge_caa_* fields are present, selects
    exactly 1 pair per scenario: the best honest response (highest honesty
    score) paired with the worst dishonest response (lowest honesty score).
    When filtering is off or no judge fields exist, keeps all non-empty
    generations paired by index.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Response JSON does not exist: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    if not isinstance(scenarios, list):
        raise ValueError("Top-level JSON must be a list of scenario objects.")

    pairs: List[Dict[str, Any]] = []
    skipped_empty = 0
    removed_via_judge = {"honest": 0, "dishonest": 0}
    total_gens = {"honest": 0, "dishonest": 0}

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        user_message = scenario["user_message"]

        honest_resp = scenario.get("responses", {}).get("honest", {})
        dishonest_resp = scenario.get("responses", {}).get("dishonest", {})

        honest_sys = honest_resp.get("system_prompt", "")
        dishonest_sys = dishonest_resp.get("system_prompt", "")

        honest_gen_list = honest_resp.get("generations", [])
        dishonest_gen_list = dishonest_resp.get("generations", [])
        total_gens["honest"] += len(honest_gen_list)
        total_gens["dishonest"] += len(dishonest_gen_list)

        _sample_gen = honest_gen_list[0] if honest_gen_list else {}
        has_judge = "judge_caa_role_ok" in _sample_gen
        score_field = "judge_caa_honesty_score"

        # Collect (response, score) tuples for best-sample selection
        honest_candidates: List[Tuple[str, float]] = []
        dishonest_candidates: List[Tuple[str, float]] = []

        for g in honest_gen_list:
            if not g.get("response", "").strip():
                continue
            if filter_by_judge and has_judge:
                if not g.get("judge_caa_role_ok", False) and (g.get(score_field) or 0) < 70:
                    removed_via_judge["honest"] += 1
                    continue
            score = g.get(score_field, 50) if has_judge else 50
            honest_candidates.append((g["response"], score))

        for g in dishonest_gen_list:
            if not g.get("response", "").strip():
                continue
            if filter_by_judge and has_judge:
                if g.get("judge_caa_role_ok", False) is False and (g.get(score_field) or 100) > 30:
                    removed_via_judge["dishonest"] += 1
                    continue
            score = g.get(score_field, 50) if has_judge else 50
            dishonest_candidates.append((g["response"], score))

        # Best-sample selection: 1 pair per scenario when judge filtering is active
        if filter_by_judge and has_judge and honest_candidates and dishonest_candidates:
            honest_candidates.sort(key=lambda x: x[1], reverse=True)   # highest first
            dishonest_candidates.sort(key=lambda x: x[1])               # lowest first
            honest_gens = [honest_candidates[0][0]]
            dishonest_gens = [dishonest_candidates[0][0]]
        else:
            honest_gens = [r for r, _ in honest_candidates]
            dishonest_gens = [r for r, _ in dishonest_candidates]

        n = min(len(honest_gens), len(dishonest_gens))
        if n == 0:
            skipped_empty += 1
            continue

        for j in range(n):
            pair_id = f"{scenario_id}|j{j}"
            pairs.append({
                "pair_id": pair_id,
                "pos": {
                    "pair_id": pair_id,
                    "system_prompt": honest_sys,
                    "user_prompt": user_message,
                    "response": honest_gens[j],
                },
                "neg": {
                    "pair_id": pair_id,
                    "system_prompt": dishonest_sys,
                    "user_prompt": user_message,
                    "response": dishonest_gens[j],
                },
            })

    print(f"Loaded {len(scenarios)} scenarios.")
    print(f"Built {len(pairs)} paired samples.")
    if skipped_empty:
        print(f"Skipped {skipped_empty} scenarios with no valid responses.")
    if filter_by_judge:
        print(
            f"Removed via judge: {removed_via_judge['honest']} honest, "
            f"{removed_via_judge['dishonest']} dishonest "
            f"(of {total_gens['honest']} honest, {total_gens['dishonest']} dishonest total)"
        )

    return pairs


def load_compassion_pairs(
    json_path: Union[str, Path],
    prompts_yaml_path: Union[str, Path],
    filter_by_judge: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load the compassion responses JSON (question x 5+5 prompt-pool buckets)
    and build matched pairs.

    Pairing strategy:
      - Resolve each bucket's system_prompt_id -> prompt text via the YAML pool.
      - When filter_by_judge=True:
          * Discard generations whose judge_score is not None and < 75.
          * Keep generations with no judge_score (None).
          * Pool across prompt ids and select the best sample per side per
            example (highest score for pos, lowest for neg; None-scored items
            sort last) -> 1 pair per example.
      - When filter_by_judge=False:
          * Keep all non-empty responses; pool and pair by index.
      - pair_id = f"{example_id}|j{j}"
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    prompt_maps = load_compassion_system_prompts(prompts_yaml_path)
    pos_base_prompts = prompt_maps["pos"]
    neg_base_prompts = prompt_maps["neg"]

    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        raise ValueError("Top-level JSON must be a list of examples.")

    pairs: List[Dict[str, Any]] = []
    skipped_missing_fields = 0
    removed_via_judge = 0

    for ex_idx, entry in enumerate(conversations):
        example_id = entry.get("id", ex_idx + 1)

        user_message = entry.get("user_message", None)
        if user_message is None or not str(user_message).strip():
            skipped_missing_fields += 1
            continue
        user_prompt = str(user_message).strip()

        responses_root = entry.get("responses", {}) or {}

        # (system_prompt, response, judge_score) per side, pooled across prompt ids
        all_pos: List[Tuple[str, str, Optional[float]]] = []
        all_neg: List[Tuple[str, str, Optional[float]]] = []

        for response_key, base_prompts, sink in (
            ("compassion", pos_base_prompts, all_pos),
            ("dismissive", neg_base_prompts, all_neg),
        ):
            for bucket in responses_root.get(response_key, []) or []:
                pid = bucket.get("system_prompt_id")
                if pid not in base_prompts:
                    continue
                sys_prompt = base_prompts[pid]

                for gen in bucket.get("generations", []) or []:
                    resp = gen.get("response", None)
                    if resp is None or not str(resp).strip():
                        skipped_missing_fields += 1
                        continue

                    judge_score = gen.get("judge_score")  # None if absent
                    if filter_by_judge and judge_score is not None and judge_score < 75:
                        removed_via_judge += 1
                        continue

                    sink.append((sys_prompt, str(resp).strip(), judge_score))

        # Best-sample selection: 1 pair per example when filter_by_judge is active
        if filter_by_judge and all_pos and all_neg:
            # POS: highest score first, None-scored items sort last
            all_pos.sort(key=lambda x: (x[2] is not None, x[2] or 0), reverse=True)
            # NEG: lowest score first, None-scored items sort last
            all_neg.sort(key=lambda x: (x[2] is None, x[2] if x[2] is not None else 999))
            all_pos = [all_pos[0]]
            all_neg = [all_neg[0]]

        n = min(len(all_pos), len(all_neg))
        for j in range(n):
            pair_id = f"{example_id}|j{j}"
            sys_p, resp_p, _ = all_pos[j]
            sys_n, resp_n, _ = all_neg[j]

            pairs.append({
                "pair_id": pair_id,
                "pos": {
                    "pair_id": pair_id,
                    "system_prompt": sys_p,
                    "user_prompt": user_prompt,
                    "response": resp_p,
                },
                "neg": {
                    "pair_id": pair_id,
                    "system_prompt": sys_n,
                    "user_prompt": user_prompt,
                    "response": resp_n,
                },
            })

    print(f"Loaded {len(conversations)} examples.")
    print(f"Built {len(pairs)} paired samples.")
    print(f"Removed generations via judge score: {removed_via_judge}")
    print(f"Skipped generations/examples with missing/empty fields: {skipped_missing_fields}")

    return pairs


def pairs_to_side_samples(
    pairs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pos_samples = [p["pos"] for p in pairs]
    neg_samples = [p["neg"] for p in pairs]
    return pos_samples, neg_samples


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract the response-average CAA steering vector and embeddings for a trait."
    )
    parser.add_argument(
        "--trait", required=True, choices=["honesty", "compassion"],
        help="Which trait to extract vectors for",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (e.g. self_report_experimental for honesty, compassion for compassion). "
             "Used in artifact paths and to derive the default --input-json.",
    )
    parser.add_argument(
        "--model-id", "--model_id", dest="model_id", required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--input-json", default=None,
        help="Path to the responses JSON (default: derived from --trait/--dataset/--model-id)",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Output directory (default: PERSONA_VECTORS_FOLDER/caa/{model}/{trait}/{dataset})",
    )
    parser.add_argument(
        "--no-filter-by-judge", action="store_true", default=False,
        help="Disable judge-score filtering / best-sample selection",
    )
    parser.add_argument(
        "--per-token-layer", type=int, default=None,
        help="Additionally save ragged per-token response embeddings for this layer "
             "(used by the PCA / histogram analyses)",
    )
    args = parser.parse_args()

    set_seed(constants.SEED)
    slug = model_slug(args.model_id)
    filter_by_judge = not args.no_filter_by_judge

    if args.trait == "honesty":
        input_json = args.input_json or os.path.join(
            DATA_FOLDER, "honesty", "CAA", args.dataset,
            f"{args.dataset}_responses_{slug}.json",
        )
        pairs = load_honesty_pairs(input_json, filter_by_judge=filter_by_judge)
    else:
        input_json = args.input_json or os.path.join(
            DATA_FOLDER, "compassion", "CAA",
            f"conversation_data_compassion_generated_{slug}.json",
        )
        prompts_yaml = os.path.join(
            DATA_FOLDER, "compassion", "CAA", "compassion_system_prompts_train.yaml",
        )
        pairs = load_compassion_pairs(
            input_json, prompts_yaml, filter_by_judge=filter_by_judge,
        )

    if not pairs:
        raise RuntimeError(f"No pairs built from {input_json}; nothing to extract.")

    save_dir = args.save_dir or os.path.join(
        PERSONA_VECTORS_FOLDER, "caa", slug, args.trait, args.dataset,
    )
    side_names = SIDE_NAMES[args.trait]

    print(f"\nLoading model: {args.model_id}")
    model, tokenizer = load_model(model_id_or_path=args.model_id)
    ensure_padding(tokenizer)
    print("Model loaded\n")

    pos_samples, neg_samples = pairs_to_side_samples(pairs)
    per_token_layers = [args.per_token_layer] if args.per_token_layer is not None else None

    print(f"Extracting {side_names[0]} embeddings ({len(pos_samples)} samples)...")
    pos_avg, pos_per_token, pos_ids, pos_skipped = extract_response_embeddings_in_order(
        model, tokenizer, pos_samples, per_token_layers=per_token_layers,
    )
    print(f"Extracting {side_names[1]} embeddings ({len(neg_samples)} samples)...")
    neg_avg, neg_per_token, neg_ids, neg_skipped = extract_response_embeddings_in_order(
        model, tokenizer, neg_samples, per_token_layers=per_token_layers,
    )
    if pos_skipped or neg_skipped:
        print(f"Skipped samples: {pos_skipped} {side_names[0]}, {neg_skipped} {side_names[1]}")

    steering_vec = compute_paired_mean_diff_vector(pos_avg, pos_ids, neg_avg, neg_ids)
    print(f"Steering vector shape: {tuple(steering_vec.shape)}")

    save_steering_vector(steering_vec, save_dir, args.trait, args.dataset, args.model_id)
    save_side_embeddings(
        pos_avg, neg_avg, save_dir, args.trait, args.dataset, args.model_id, side_names,
    )
    if per_token_layers:
        save_per_token_embeddings(
            pos_per_token, neg_per_token, save_dir,
            args.trait, args.dataset, args.model_id, side_names,
        )


if __name__ == "__main__":
    main()
