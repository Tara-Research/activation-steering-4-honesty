"""AlpacaEval 2.0 — Generation Phase.

Generates steered and unsteered responses on the 805 AlpacaEval prompts.
Uses the same steering hooks as the main experiments (SteeringWithFixedCoef,
SteeringToTargetProjection, SteeringToMirrorProjection) applied via context managers during
``model.generate()``.

max_tokens set to 2048 to avoid truncation

Config is loaded dynamically via ``constants.get_config_for_trait()``.

Output format (one file per technique)::

    Baseline:
    {
      "metadata": { ..., "technique": "baseline_no_steering" },
      "items": [ {"instruction": ..., "output": ..., "output_tokens": ...} ]
    }

    Steering:
    {
      "metadata": { ..., "technique": "swfc" },
      "items": { "<coef>": [ ... ], ... }
    }
"""

import argparse
import gc
import json
import os

import torch
import yaml

from tqdm import trange

from src.constants import constants
from src.constants.constants import PERSONA_VECTORS_FOLDER
from src.steering.sttp import SteeringToTargetProjection
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.stmp import SteeringToMirrorProjection
from src.utils.model_utils import load_model

MODE_TO_TYPE = {
    "all_token_mode": "all",
    "response_mode": "response",
}

ALL_TECHNIQUES = [
    "baseline_no_steering",
    "swfc",
    "sttp",
    "stmp",
]

STEERING_TECHNIQUES = [
    "swfc",
    "sttp",
    "stmp",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_alpaca_eval_prompts(test: bool = False):
    """Return (instructions, datasets) from the AlpacaEval dataset.

    When *test* is True, take the first prompt from each source dataset
    category (the ``dataset`` field) for a quick smoke test.
    """
    alpaca_eval_path = os.path.join(constants.DATA_FOLDER, "alpaca_eval", "alpaca_eval.json")
    with open(alpaca_eval_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    if test:
        from collections import defaultdict
        per_cat = defaultdict(list)
        for row in ds:
            cat = row["dataset"]
            if len(per_cat[cat]) < 5:
                per_cat[cat].append(row)
        rows = [row for rows in per_cat.values() for row in rows]
        return [r["instruction"] for r in rows], [r["dataset"] for r in rows]
    return [row["instruction"] for row in ds], [row["dataset"] for row in ds]


def _build_conversations(instructions):
    """Wrap each instruction into a single-turn conversation list."""
    return [[{"role": "user", "content": inst}] for inst in instructions]


def _generate_batch(model, tokenizer, conversations, *, max_tokens, temperature,
                    top_p, min_tokens, bs):
    """Baseline generation (no steering) matching ``steering_sampling.sample``."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in conversations
    ]

    outputs = []
    output_token_counts = []
    for i in trange(0, len(prompts), bs, desc="Generating (baseline)"):
        batch = prompts[i:i + bs]
        tok_batch = tokenizer(batch, return_tensors="pt", padding=True)
        device = model.get_input_embeddings().weight.device
        tok_batch = {k: v.to(device, non_blocking=True) for k, v in tok_batch.items()}

        with torch.no_grad():
            generated = model.generate(
                **tok_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                use_cache=True,
            )

        prompt_len = tok_batch["input_ids"].shape[1]
        for seq in generated:
            new_tokens = seq[prompt_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
            output_token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))

        del generated, tok_batch
        gc.collect()
        torch.cuda.empty_cache()

    return outputs, output_token_counts


def _generate_steered_batch(model, tokenizer, conversations, steerer_ctx, *,
                            max_tokens, temperature, top_p, min_tokens, bs):
    """Generation with an active steerer context manager."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in conversations
    ]

    outputs = []
    output_token_counts = []
    device = model.get_input_embeddings().weight.device
    with steerer_ctx:
        for i in trange(0, len(prompts), bs, desc="Generating (steered)"):
            batch = prompts[i:i + bs]
            tok_batch = tokenizer(batch, return_tensors="pt", padding=True)
            tok_batch = {k: v.to(device, non_blocking=True) for k, v in tok_batch.items()}

            with torch.no_grad():
                generated = model.generate(
                    **tok_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    min_new_tokens=min_tokens,
                    use_cache=True,
                )

            prompt_len = tok_batch["input_ids"].shape[1]
            for seq in generated:
                new_tokens = seq[prompt_len:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                outputs.append(text)
                output_token_counts.append(len(tokenizer.encode(text, add_special_tokens=False)))

            del generated, tok_batch
            gc.collect()
            torch.cuda.empty_cache()

    return outputs, output_token_counts


def _pack_results(instructions, datasets, texts, token_counts):
    return [
        {"dataset": ds, "instruction": inst, "output": txt, "output_tokens": ntok}
        for ds, inst, txt, ntok in zip(datasets, instructions, texts, token_counts)
    ]


def _build_steerer(technique, model, steering_vectors, neg_embeddings,
                   pos_embeddings, coef, layer_idx, positions, bias=None):
    """Instantiate the appropriate steerer context manager."""
    if technique == "swfc":
        vector = steering_vectors[layer_idx]
        return SteeringWithFixedCoef(
            model, vector, coeff=coef, layer_idx=layer_idx, positions=positions,
        )
    elif technique == "sttp":
        return SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            coefficient=coef,
            layer_idx=layer_idx,
            positions=positions,
            negative_embeddings=neg_embeddings,
            positive_embeddings=pos_embeddings,
            bias=bias,
        )
    elif technique == "stmp":
        return SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            coefficient=coef,
            layer_idx=layer_idx,
            positions=positions,
            negative_embeddings=neg_embeddings,
            positive_embeddings=pos_embeddings,
            bias=bias,
        )
    else:
        raise ValueError(f"Unknown steering technique: {technique}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlpacaEval 2.0 — generate steered/unsteered responses"
    )
    parser.add_argument("--model-id", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--trait", type=str, default="compassion")
    parser.add_argument("--steering-mode", type=str, default="all_token_mode",
                        choices=list(MODE_TO_TYPE),
                        help="Steering mode: 'all_token_mode' or 'response_mode'")
    parser.add_argument("--honesty-category", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 1 prompt per dataset category")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (used in result directory)")
    parser.add_argument("--techniques", nargs="+",
                        choices=ALL_TECHNIQUES,
                        default=ALL_TECHNIQUES,
                        help="Techniques to run (default: all four)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output file exists")
    parser.add_argument("--coefficients", type=float, nargs="+", default=None,
                        help="Override the steering coefficients (target s = mu_pos + "
                             "coef*std_pos) for ALL requested steering techniques, instead "
                             "of all_layers.<technique>.coefficients. Intended for "
                             "single-technique runs (e.g. the threshold-alpha sweep).")
    args = parser.parse_args()

    model_id = args.model_id
    trait = args.trait
    steering_type = MODE_TO_TYPE[args.steering_mode]
    honesty_category = args.honesty_category
    requested_techniques = args.techniques

    if trait == "honesty" and not honesty_category:
        parser.error("--honesty-category is required when --trait is 'honesty'")

    model_save_name = model_id.replace("/", "_")
    config_file = constants.get_config_for_trait(trait)

    # ---- Load config ----
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    gen_defaults = cfg["generation_defaults"]
    max_tokens = 2048  # AlpacaEval needs longer outputs than default config
    gen_defaults["max_tokens"] = max_tokens
    temperature = gen_defaults["temperature"]
    top_p = gen_defaults["top_p"]
    min_tokens = gen_defaults["min_tokens"]
    bs = gen_defaults["batch_size"]

    # ---- Resolve best layer from config if not overridden ----
    # Only honesty configs partition `best_layer_*` by category; compassion
    # uses a single model-keyed block.
    if honesty_category:
        best_layer_key = f"best_layer_{model_save_name}_{honesty_category}"
    else:
        best_layer_key = f"best_layer_{model_save_name}"
    best_layer_cfg = cfg[best_layer_key][args.steering_mode]
    # Only resolve layers for the steering techniques actually requested; the
    # config block for a given (model, steering_mode) need not define every
    # technique.
    requested_steering = [t for t in requested_techniques if t in STEERING_TECHNIQUES]
    layer_per_technique = {t: best_layer_cfg[t]["layer"] for t in requested_steering}
    print(f"Using best layers from config: {layer_per_technique}")

    # ---- Coefficients: take middle 3 from all_layers to keep benchmark lean ----
    all_layers_cfg = cfg["all_layers"]
    # def _middle_3(coefs):
    #     """Return the 3 middle coefficients from a sorted list."""
    #     if len(coefs) <= 3:
    #         return coefs
    #     start = (len(coefs) - 3) // 2
    #     return coefs[start:start + 3]

    if args.coefficients is not None:
        # Override applies to every requested steering technique (intended for
        # single-technique runs such as the threshold-alpha sweep).
        coefficients = {t: args.coefficients for t in requested_steering}
    else:
        coefficients = {
            t: all_layers_cfg[t]["coefficients"] for t in requested_steering
        }
    print(f"Coefficients: {coefficients}")

    # ---- Load steering artefacts (only if any steering technique is requested) ----
    steering_vectors = None
    neg_embeddings = None
    pos_embeddings = None
    bias = None

    if requested_steering:
        if honesty_category:
            common_cfg = cfg[f"common_steering_config_{model_save_name}_{honesty_category}"]
        else:
            common_cfg = cfg[f"common_steering_config_{model_save_name}"]

        common_cfg["vector_path"] = os.path.join(
            PERSONA_VECTORS_FOLDER, common_cfg["vector_path"]
        )
        common_cfg["neg_embeddings_path"] = os.path.join(
            PERSONA_VECTORS_FOLDER, common_cfg["neg_embeddings_path"]
        )
        common_cfg["pos_embeddings_path"] = os.path.join(
            PERSONA_VECTORS_FOLDER, common_cfg["pos_embeddings_path"]
        )

        if common_cfg.get("bias_path"):
            common_cfg["bias_path"] = os.path.join(
                PERSONA_VECTORS_FOLDER, common_cfg["bias_path"]
            )
            bias = torch.load(common_cfg["bias_path"], weights_only=False)
            print(f"Loaded LR bias from {common_cfg['bias_path']}")

        steering_vectors = torch.load(common_cfg["vector_path"], weights_only=False)
        neg_embeddings = torch.load(common_cfg["neg_embeddings_path"])
        pos_embeddings = torch.load(common_cfg["pos_embeddings_path"])

        bias = None
        if common_cfg.get("bias_path"):
            bias_full_path = os.path.join(PERSONA_VECTORS_FOLDER, common_cfg["bias_path"])
            bias = torch.load(bias_full_path, weights_only=False)

        if isinstance(pos_embeddings, dict) or isinstance(neg_embeddings, dict):
            raise ValueError(
                "per-token (ragged) embeddings are no longer supported; "
                "expected dense [num_layers, N, D] response_avg tensors"
            )

    # ---- Load model ----
    model, tokenizer = load_model(model_id)
    model.eval()

    # ---- Load AlpacaEval prompts ----
    instructions, datasets = _load_alpaca_eval_prompts(test=args.test)
    conversations = _build_conversations(instructions)
    print(f"Loaded {len(instructions)} AlpacaEval prompts")

    gen_kwargs = dict(
        max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, min_tokens=min_tokens, bs=bs,
    )

    exp_name = args.exp_name

    base_metadata = {
        "model_id": model_id,
        "layer_per_technique": layer_per_technique,
        "trait": trait,
        "vector_type": constants.VECTOR_TYPE,
        "steering_mode": args.steering_mode,
        "steering_type": steering_type,
        "exp_name": exp_name,
        "test_mode": args.test,
        "num_prompts": len(instructions),
        "generation_defaults": gen_defaults,
    }

    output_dir = os.path.join(
        constants.ALPACA_EVAL_RESULTS_DIR, constants.VECTOR_TYPE, trait,
        model_save_name, exp_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    # ---- Run each requested technique ----
    for technique in requested_techniques:
        metadata = {**base_metadata, "technique": technique}
        out_path = os.path.join(
            output_dir, f"generations_{steering_type}_{technique}.json"
        )

        if technique == "baseline_no_steering":
            # Resume check: skip if baseline already complete
            if os.path.exists(out_path) and not args.force:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if len(existing.get("items", [])) == len(instructions):
                    print(f"=== Baseline already complete ({len(instructions)} items), skipping ===")
                    continue

            print("=== Baseline (no steering) ===")
            base_texts, base_toks = _generate_batch(
                model, tokenizer, conversations, **gen_kwargs
            )
            results = {
                "metadata": metadata,
                "items": _pack_results(instructions, datasets, base_texts, base_toks),
            }
            print(f"  avg tokens: {sum(base_toks) / len(base_toks):.1f}")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Saved to {out_path}")

        else:
            # Resume: load existing results if present
            if os.path.exists(out_path) and not args.force:
                with open(out_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                done_coefs = set(results["items"].keys())
                print(f"=== {technique}: resuming, {len(done_coefs)} coefficients already done ===")
            else:
                results = {"metadata": metadata, "items": {}}
                done_coefs = set()

            for coef in coefficients[technique]:
                if str(coef) in done_coefs:
                    print(f"  Skipping coef={coef} (already completed)")
                    continue

                print(f"=== {technique}, coef={coef} ===")
                ctx = _build_steerer(
                    technique, model, steering_vectors, neg_embeddings,
                    pos_embeddings, coef,
                    layer_idx=layer_per_technique[technique],
                    positions=steering_type,
                    bias=bias,
                )
                texts, toks = _generate_steered_batch(
                    model, tokenizer, conversations, ctx, **gen_kwargs
                )
                results["items"][str(coef)] = _pack_results(instructions, datasets, texts, toks)
                print(f"  avg tokens: {sum(toks) / len(toks):.1f}")

                # Incremental save after each coefficient
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  Saved to {out_path}")
