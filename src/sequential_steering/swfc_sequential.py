import json
from typing import Any, Dict, List, Set

from src.steering.steering_sampling import (
    sample_swfc,
    sample_multi_layer_swfc,
)


def _rows_from_answers(conversations, answers, has_sys_prompt: bool, scenario_meta=None):
    rows = []
    meta = scenario_meta or [{}] * len(conversations)
    for conv, ans, m in zip(conversations, answers, meta):
        q = conv[1]["content"] if has_sys_prompt else conv[0]["content"]
        rows.append({
            "scenario_id": m.get("scenario_id"),
            "category": m.get("category"),
            "question": q,
            "steered_answer": ans,
        })
    return rows


def run_all_layers_swfc_sequential(
    *,
    model,
    tokenizer,
    conversations,
    steering_vectors,
    neg_embeddings,   # unused; kept for uniform interface with wrapper
    pos_embeddings,   # unused; kept for uniform interface with wrapper
    common_steering_config: Dict[str, Any],
    specific_steering_config: Dict[str, Any],
    all_layers: List[int],
    done_keys: Set[str],
    experiment_results: Dict[str, Any],
    output_json: str,
    bias=None,        # unused; kept for uniform interface with wrapper
    scenario_meta=None,
) -> None:
    has_sys_prompt = bool(specific_steering_config["sys_prompt"])

    steering_type = specific_steering_config["steering_type"]
    enable_thinking = bool(specific_steering_config["enable_thinking"])

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for messages in conversations
    ]

    coefs = specific_steering_config["coefficients"]

    for layer in all_layers:
        layer_key = str(layer)
        existing = experiment_results["layers"].get(layer_key, {})
        coefs_to_run = [c for c in coefs if f"coef_{c}" not in existing]
        if not coefs_to_run:
            continue

        print(f"[steer] layer {layer} (fixed steering, coef sweep) — {len(coefs_to_run)} new coef(s)")

        results_for_layer: Dict[str, Any] = dict(existing)
        for coef in coefs_to_run:
            answers = sample_swfc(
                model=model,
                tokenizer=tokenizer,
                conversations=conversations,
                prompts=prompts,
                vector=steering_vectors[layer],
                layer=layer,
                coef=coef,
                bs=common_steering_config["batch_size"],
                top_p=common_steering_config["top_p"],
                max_tokens=common_steering_config["max_tokens"],
                temperature=common_steering_config["temperature"],
                min_tokens=common_steering_config["min_tokens"],
                steering_type=steering_type,
                enable_thinking=enable_thinking,
            )

            results_for_layer[f"coef_{coef}"] = _rows_from_answers(conversations, answers, has_sys_prompt, scenario_meta)

        experiment_results["layers"][layer_key] = results_for_layer

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, ensure_ascii=False, indent=2)


def run_all_layers_swfc_multiple_layers_sequential(
    *,
    model,
    tokenizer,
    conversations,
    steering_vectors,
    neg_embeddings,   # unused; kept for uniform interface with wrapper
    pos_embeddings,   # unused; kept for uniform interface with wrapper
    common_steering_config: Dict[str, Any],
    specific_steering_config: Dict[str, Any],
    all_layers: List[int],
    done_keys: Set[str],
    experiment_results: Dict[str, Any],
    output_json: str,
    bias=None,        # unused; kept for uniform interface with wrapper
    scenario_meta=None,
) -> None:
    has_sys_prompt = bool(specific_steering_config["sys_prompt"])

    steering_type = specific_steering_config["steering_type"]
    enable_thinking = bool(specific_steering_config["enable_thinking"])

    coefs = specific_steering_config["coefficients"]

    for layer in all_layers:
        if layer > 31:
            break

        window_key = f"{layer}_{layer + 5}"
        if window_key in done_keys:
            continue

        print(f"[steer] window {window_key} (fixed multi-layer, coef sweep)")

        layers_for_steering = [l for l in range(layer, layer + 5)]
        results_for_window: Dict[str, Any] = {}

        for coefficient in coefs:
            answers = sample_multi_layer_swfc(
                model=model,
                tokenizer=tokenizer,
                conversations=conversations,
                vectors=steering_vectors,
                layers=layers_for_steering,
                coefficient=float(coefficient),
                bs=common_steering_config["batch_size"],
                top_p=common_steering_config["top_p"],
                max_tokens=common_steering_config["max_tokens"],
                temperature=common_steering_config["temperature"],
                min_tokens=common_steering_config["min_tokens"],
                steering_type=steering_type,
                enable_thinking=enable_thinking,
            )

            # keep your key naming (note: will be "coef_0.2" etc.)
            results_for_window[f"coef_{coefficient}"] = _rows_from_answers(conversations, answers, has_sys_prompt, scenario_meta)

        experiment_results["layers"][window_key] = results_for_window

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, ensure_ascii=False, indent=2)
