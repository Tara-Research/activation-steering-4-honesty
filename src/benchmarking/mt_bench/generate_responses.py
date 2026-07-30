"""Stage 1: Generate steered/unsteered MT-Bench answers.

Usage:
    python -m src.mt_bench.generate_responses \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait honesty \
        --techniques no_steering swfc
"""

import argparse
import contextlib
import json
import os
import re
from typing import List, Optional

import torch
import yaml

from src.constants.constants import (
    MT_BENCH_DATA_DIR,
    MT_BENCH_RESULTS_DIR,
    CONFIG_MT_BENCH,
    PERSONA_VECTORS_FOLDER,
    VECTOR_TYPE,
    get_config_for_trait,
)
from src.benchmarking.mt_bench.data_utils import BASELINE_TECHNIQUES, filter_one_per_category, load_questions
from src.multi_turn_steering.generate_answer import generate_steered_answer, generate_plain
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.sttp import SteeringToTargetProjection
from src.steering.stmp import SteeringToMirrorProjection
from src.utils.model_utils import load_model

_STEERING_MODE_TO_TYPE = {
    "all_token_mode": "all",
    "response_mode": "response",
}


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 reasoning traces)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _build_output_dir(model_id: str, trait: str, exp_name: str) -> str:
    model_save_name = model_id.replace("/", "_")
    return os.path.join(MT_BENCH_RESULTS_DIR, VECTOR_TYPE, trait, model_save_name, exp_name)


def _output_path(output_dir: str, technique: str) -> str:
    return os.path.join(output_dir, f"model_answer_{technique}.jsonl")


def _load_coefficients(trait_cfg, technique):
    """Return list of coefficient floats for a steering technique, or empty list for baselines."""
    if technique in BASELINE_TECHNIQUES:
        return []
    coefficients = trait_cfg["all_layers"][technique]["coefficients"]
    return [float(c) for c in coefficients]


def _load_completed_ids(output_file: str) -> set:
    """Load (question_id, coefficient) pairs already present in an output JSONL for resume."""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed.add((rec["question_id"], rec.get("coefficient")))
    return completed


def _build_steering_cfg(args, trait_cfg, trait_config_path):
    """Build a flat steering config dict from trait YAML + MT-Bench generation config.

    Merges:
      1. MT-Bench generation defaults
      2. common_steering_config (vector paths)
      3. best_layer config (layer, coefficient)
      4. CLI overrides
    """
    model_save_name = args.model_id.replace("/", "_")

    if args.trait == "honesty":
        common_key = f"common_steering_config_{model_save_name}_{args.honesty_category}"
    else:
        common_key = f"common_steering_config_{model_save_name}"
    common_cfg = trait_cfg.get(common_key, {})

    gen_cfg = trait_cfg["generation_defaults"]

    is_steering = args.technique not in BASELINE_TECHNIQUES

    if is_steering and common_key not in trait_cfg:
        raise KeyError(
            f"Missing required config section: '{common_key}' in {trait_config_path}"
        )

    best_layer_cfg = {}
    if is_steering:
        if args.trait == "honesty":
            best_layer_key = f"best_layer_{model_save_name}_{args.honesty_category}"
        else:
            best_layer_key = f"best_layer_{model_save_name}"

        if best_layer_key not in trait_cfg:
            raise KeyError(
                f"Missing required config section: '{best_layer_key}' in {trait_config_path}"
            )
        best_layers = trait_cfg[best_layer_key]
        if args.steering_mode not in best_layers:
            raise KeyError(
                f"Missing steering mode '{args.steering_mode}' in config section "
                f"'{best_layer_key}'"
            )
        mode_layers = best_layers[args.steering_mode]
        if args.technique not in mode_layers:
            raise KeyError(
                f"Missing technique '{args.technique}' in config section "
                f"'{best_layer_key}.{args.steering_mode}'"
            )
        best_layer_cfg = mode_layers[args.technique]

    return {
        "steering_technique": args.technique,
        "max_tokens": gen_cfg["max_tokens"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
        "enable_thinking": gen_cfg["enable_thinking"],
        "steering_direction": "positive",
        "steering_type": _STEERING_MODE_TO_TYPE[args.steering_mode],
        "layer": best_layer_cfg["layer"] if is_steering else None,
        "layers": None,
        "coefficient": best_layer_cfg["coefficient"] if is_steering else 1.0,
        "vector_path": common_cfg["vector_path"] if is_steering else None,
        "neg_embeddings_path": common_cfg["neg_embeddings_path"],
        "pos_embeddings_path": common_cfg["pos_embeddings_path"],
        "bias_path": common_cfg.get("bias_path"),
    }


def _load_steering_artifacts(steering_cfg):
    """Load steering vectors / embeddings from the paths in steering_cfg."""
    steering_vectors = None
    neg_embeddings = None
    pos_embeddings = None

    vec_path = steering_cfg["vector_path"]
    neg_path = steering_cfg["neg_embeddings_path"]
    pos_path = steering_cfg["pos_embeddings_path"]

    if vec_path:
        full_path = os.path.join(PERSONA_VECTORS_FOLDER, vec_path)
        steering_vectors = torch.load(full_path, map_location="cpu", weights_only=False)
    if neg_path:
        full_path = os.path.join(PERSONA_VECTORS_FOLDER, neg_path)
        neg_embeddings = torch.load(full_path, map_location="cpu", weights_only=False)
    if pos_path:
        full_path = os.path.join(PERSONA_VECTORS_FOLDER, pos_path)
        pos_embeddings = torch.load(full_path, map_location="cpu", weights_only=False)

    bias = None
    bias_path = steering_cfg.get("bias_path")
    if bias_path:
        full_path = os.path.join(PERSONA_VECTORS_FOLDER, bias_path)
        bias = torch.load(full_path, map_location="cpu", weights_only=False)

    if isinstance(pos_embeddings, dict) or isinstance(neg_embeddings, dict):
        raise ValueError(
            "per-token (ragged) embeddings are no longer supported; "
            "expected dense [num_layers, N, D] response_avg tensors"
        )

    return steering_vectors, neg_embeddings, pos_embeddings, bias


def _create_steerer_ctx(model, steering_cfg, steering_vectors, neg_embeddings, pos_embeddings, bias=None):
    """Return a context manager that activates the appropriate steerer (or a no-op for baselines)."""
    technique = steering_cfg["steering_technique"]

    if technique in BASELINE_TECHNIQUES:
        return contextlib.nullcontext()

    layer = steering_cfg["layer"]
    coefficient = steering_cfg["coefficient"]
    steering_type = steering_cfg["steering_type"]
    steering_direction = steering_cfg["steering_direction"]

    if technique == "swfc":
        return SteeringWithFixedCoef(
            model,
            steering_vector=steering_vectors[layer],
            coeff=coefficient,
            layer_idx=layer,
            positions=steering_type,
        )
    elif technique == "sttp":
        return SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=neg_embeddings,
            positive_embeddings=pos_embeddings,
            layer_idx=layer,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
        )
    elif technique == "stmp":
        return SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=neg_embeddings,
            positive_embeddings=pos_embeddings,
            layer_idx=layer,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
        )
    else:
        raise ValueError(f"Unknown steering technique: {technique}")


def _generate_batch(
    model,
    tokenizer,
    conversations: List[List[dict]],
    steering_cfg: dict,
    steering_vectors: Optional[torch.Tensor],
    neg_embeddings: Optional[torch.Tensor],
    pos_embeddings: Optional[torch.Tensor],
    batch_size: int,
    bias: Optional[torch.Tensor] = None,
) -> List[str]:
    """Generate responses for all conversations in batched passes.

    The steerer context is entered once and wraps all sub-batches so that
    distribution/mirror steerers compute projection stats only once.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=steering_cfg["enable_thinking"],
        )
        for msgs in conversations
    ]

    steerer_ctx = _create_steerer_ctx(
        model, steering_cfg, steering_vectors, neg_embeddings, pos_embeddings, bias=bias,
    )

    temperature = steering_cfg["temperature"]
    top_p = steering_cfg["top_p"]
    max_tokens = steering_cfg["max_tokens"]
    input_device = model.get_input_embeddings().weight.device

    outputs: List[str] = []
    with steerer_ctx:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            tokenized = tokenizer(batch, return_tensors="pt", padding=True)
            tokenized = {k: v.to(input_device, non_blocking=True) for k, v in tokenized.items()}

            with torch.no_grad():
                generated = model.generate(
                    **tokenized,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                )

            prompt_len = tokenized["input_ids"].shape[1]
            decoded = [
                tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                for seq in generated
            ]
            outputs.extend(decoded)
            del generated, tokenized

    return outputs


def _generate_one(model, tokenizer, messages, steering_cfg,
                   steering_vectors, neg_embeddings, pos_embeddings, bias=None):
    """Generate a single response, steered or plain."""
    technique = steering_cfg["steering_technique"]

    if technique in BASELINE_TECHNIQUES:
        return generate_plain(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=steering_cfg["max_tokens"],
            temperature=steering_cfg["temperature"],
            top_p=steering_cfg["top_p"],
            enable_thinking=steering_cfg["enable_thinking"],
        )

    return generate_steered_answer(
        model=model,
        tokenizer=tokenizer,
        steering_technique=technique,
        messages=messages,
        steering_vectors=steering_vectors,
        negative_embeddings=neg_embeddings,
        positive_embeddings=pos_embeddings,
        layer=steering_cfg["layer"],
        layers=steering_cfg["layers"],
        steering_direction=steering_cfg["steering_direction"],
        coefficient=steering_cfg["coefficient"],
        steering_type=steering_cfg["steering_type"],
        max_new_tokens=steering_cfg["max_tokens"],
        temperature=steering_cfg["temperature"],
        top_p=steering_cfg["top_p"],
        enable_thinking=steering_cfg["enable_thinking"],
        bias=bias,
    )


def generate_mt_bench(
    model, tokenizer, questions, steering_cfg,
    steering_vectors, neg_embeddings, pos_embeddings,
    output_file,
    coefficient=None,
    batch_size: int = 8,
    bias=None,
):
    """Generate 2-turn MT-Bench answers for all questions using batched generation.

    Supports resume: skips (question_id, coefficient) pairs already in output_file.
    """
    completed_ids = _load_completed_ids(output_file)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} (question, coefficient) pairs already completed")

    # Filter to pending questions only
    pending = [q for q in questions if (q["question_id"], coefficient) not in completed_ids]
    if not pending:
        print("All questions already completed for this coefficient — nothing to do.")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    technique = steering_cfg["steering_technique"]

    # --- Turn 1: batch all first-turn prompts ---
    print(f"  Generating turn 1 for {len(pending)} questions (batch_size={batch_size})")
    turn1_conversations = [
        [{"role": "user", "content": q["turns"][0]}]
        for q in pending
    ]
    turn1_responses = _generate_batch(
        model, tokenizer, turn1_conversations, steering_cfg,
        steering_vectors, neg_embeddings, pos_embeddings, batch_size, bias=bias,
    )
    turn1_responses = [_strip_think_tags(r) for r in turn1_responses]

    # --- Turn 2: batch all second-turn prompts (includes turn-1 context) ---
    print(f"  Generating turn 2 for {len(pending)} questions (batch_size={batch_size})")
    turn2_conversations = [
        [
            {"role": "user", "content": q["turns"][0]},
            {"role": "assistant", "content": resp1},
            {"role": "user", "content": q["turns"][1]},
        ]
        for q, resp1 in zip(pending, turn1_responses)
    ]
    turn2_responses = _generate_batch(
        model, tokenizer, turn2_conversations, steering_cfg,
        steering_vectors, neg_embeddings, pos_embeddings, batch_size, bias=bias,
    )
    turn2_responses = [_strip_think_tags(r) for r in turn2_responses]

    # --- Write all records ---
    with open(output_file, "a", encoding="utf-8") as f:
        for q, resp1, resp2 in zip(pending, turn1_responses, turn2_responses):
            record = {
                "question_id": q["question_id"],
                "category": q["category"],
                "technique": technique,
                "coefficient": coefficient,
                "turns": [resp1, resp2],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(pending)} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MT-Bench responses with optional steering"
    )
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--trait", default="honesty", help="Trait name (honesty, compassion)")
    parser.add_argument(
        "--techniques", nargs="+", default=["stmp", "no_steering", "swfc", "sttp"],
        help="Steering techniques to evaluate",
    )
    parser.add_argument(
        "--steering_mode", default="all_token_mode",
        choices=["all_token_mode", "response_mode"],
        help="Steering mode",
    )
    parser.add_argument(
        "--honesty_category", default="self_report_experimental",
        help="Honesty category key for config lookup",
    )
    parser.add_argument("--exp_name", required=True, help="Experiment name (used in result directory)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation (default: 8)")
    parser.add_argument("--test", action="store_true", help="Test mode: 1 sample per category (8 total)")

    args = parser.parse_args()

    # Load MT-Bench config
    with open(CONFIG_MT_BENCH, "r", encoding="utf-8") as f:
        mt_bench_cfg = yaml.safe_load(f)
    mt_cfg = mt_bench_cfg["mt_bench"]

    # Load trait steering config
    trait_config_path = get_config_for_trait(args.trait)
    with open(trait_config_path, "r", encoding="utf-8") as f:
        trait_cfg = yaml.safe_load(f)

    # Load questions
    question_file = os.path.join(MT_BENCH_DATA_DIR, mt_cfg["question_file"])
    questions = load_questions(question_file)

    if args.test:
        questions = filter_one_per_category(questions)
        print(f"[TEST MODE] Using {len(questions)} questions (1 per category)")
    else:
        print(f"Loaded {len(questions)} MT-Bench questions")

    # # Load model once
    print(f"Loading model: {args.model_id}")
    model, tokenizer = load_model(args.model_id)

    output_dir = _build_output_dir(args.model_id, args.trait, args.exp_name)

    for technique in args.techniques:
        print(f"\n=== Technique: {technique} ===")
        args.technique = technique

        steering_cfg = _build_steering_cfg(args, trait_cfg, trait_config_path)

        # Load steering artifacts once per technique (vectors don't change across coefficients)
        steering_vectors, neg_embeddings, pos_embeddings, bias = _load_steering_artifacts(steering_cfg)
        if steering_vectors is not None:
            steering_vectors = steering_vectors.to(model.device)
        if neg_embeddings is not None:
            neg_embeddings = neg_embeddings.to(model.device)
        if pos_embeddings is not None:
            pos_embeddings = pos_embeddings.to(model.device)
        if bias is not None:
            bias = bias.to(model.device)

        output_file = _output_path(output_dir, technique)

        if technique in BASELINE_TECHNIQUES:
            print(f"  Baseline technique — running once with coefficient=None")
            generate_mt_bench(
                model, tokenizer, questions, steering_cfg,
                steering_vectors, neg_embeddings, pos_embeddings,
                output_file,
                coefficient=None,
                batch_size=args.batch_size,
                bias=bias,
            )
        else:
            coefficients = _load_coefficients(trait_cfg, technique)
            print(f"  Sweeping {len(coefficients)} coefficients: {coefficients}")
            for coef in coefficients:
                print(f"\n  --- Coefficient: {coef} ---")
                steering_cfg["coefficient"] = coef
                generate_mt_bench(
                    model, tokenizer, questions, steering_cfg,
                    steering_vectors, neg_embeddings, pos_embeddings,
                    output_file,
                    coefficient=coef,
                    batch_size=args.batch_size,
                    bias=bias,
                )


if __name__ == "__main__":
    main()
