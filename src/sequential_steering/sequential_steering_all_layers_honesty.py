import argparse
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from src.constants import constants
from src.constants.constants import DATA_FOLDER, RESULTS_FOLDER, PERSONA_VECTORS_FOLDER, VECTOR_TYPE, VECTOR_TYPE_LABEL
from src.sequential_steering.sttp_sequential import \
    run_all_layers_sttp_sequential, run_all_layers_sttp_multiple_layers_sequential
from src.sequential_steering.swfc_sequential import \
    run_all_layers_swfc_sequential, run_all_layers_swfc_multiple_layers_sequential
from src.sequential_steering.stmp_sequential import run_all_layers_stmp_sequential, \
    run_all_layers_stmp_multiple_layers_sequential
from src.sequential_steering.no_steering import run_all_layers_no_steering_sequential
from src.utils.model_utils import load_model, set_seed, get_num_hidden_layers
from src.utils.path_utils import build_single_turn_output_path, merge_generation_defaults


def load_conversations_honesty(path: str, aligned_baseline: bool = False):
    """
    Load honesty test scenarios with parallel scenario metadata.

    Returns:
        (conversations, scenario_meta) where conversations is a list of
        chat-message lists and scenario_meta is a parallel list of
        {"scenario_id": str, "category": str | None} dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    convos = [
        [
            {"role": "system", "content": item["system_prompt_honest"] if aligned_baseline else item["system_prompt_dishonest"]},
            {"role": "user", "content": item["user_message"]},
        ]
        for item in data["scenarios"]
    ]
    scenario_meta = [
        {"scenario_id": item["scenario_id"], "category": item["category"]}
        for item in data["scenarios"]
    ]
    return convos, scenario_meta


def _load_or_init_results(output_json: str, common_cfg: Dict[str, Any], specific_cfg: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "common_steering_config": common_cfg,
        "specific_config": specific_cfg,
        "layers": {},
    }
    
def _ensure_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id



def run_steering_experiments_honesty_all_layers_sequential(
        steering_function,
        specific_config: Dict[str, Any],
        common_steering_config: Dict[str, Any],
        aligned_baseline: bool = False,
        test_mode: bool = False,
):
    output_json = specific_config["output_json"]

    # ---- Load everything ONCE (model stays sharded across GPUs) ----
    conversations, scenario_meta = load_conversations_honesty(
        common_steering_config["data_path"],
        aligned_baseline=aligned_baseline,
    )

    hf_model, hf_tokenizer = load_model(model_id_or_path=common_steering_config["model_id"])
    _ensure_padding(hf_tokenizer)
    hf_model.config.pad_token_id = hf_tokenizer.pad_token_id
    hf_model.eval()

    steering_vectors = torch.load(common_steering_config["vector_path"], weights_only=False)
    neg_embeddings = torch.load(common_steering_config["neg_embeddings_path"])
    pos_embeddings = torch.load(common_steering_config["pos_embeddings_path"])

    bias = None
    if common_steering_config.get("bias_path"):
        bias = torch.load(common_steering_config["bias_path"], weights_only=False)

    num_layers_vec = int(steering_vectors.shape[0])
    num_layers_model = get_num_hidden_layers(hf_model)
    num_layers = min(num_layers_vec, num_layers_model)

    if isinstance(pos_embeddings, dict) or isinstance(neg_embeddings, dict):
        raise ValueError(
            "per-token (ragged) embeddings are no longer supported; "
            "expected dense [num_layers, N, D] response_avg tensors"
        )

    experiment_results = _load_or_init_results(output_json, common_steering_config, specific_config)

    # Skip completed keys (layer indices or window-keys depending on technique)
    if isinstance(experiment_results, dict):
        done_keys = set(experiment_results.get("layers", {}).keys())
    else:
        done_keys = set()

    all_layers = list(range(num_layers))

    if test_mode:
        n_test = 5
        indices = np.linspace(0, len(all_layers) - 1, n_test, dtype=int)
        all_layers = [all_layers[i] for i in indices]
        print(f"[test_mode] subsampled to {n_test} layers: {all_layers}")

    print(f"[steer] sequential run over {len(all_layers)} layers (num_layers={num_layers})")
    print("[steer] model is loaded once (should remain sharded across available GPUs)")

    set_seed(constants.SEED)

    # ---- Run sequentially; steering_function handles per-layer loop and incremental merging ----
    steering_function(
        model=hf_model,
        tokenizer=hf_tokenizer,
        conversations=conversations,
        scenario_meta=scenario_meta,
        steering_vectors=steering_vectors,
        neg_embeddings=neg_embeddings,
        pos_embeddings=pos_embeddings,
        common_steering_config=common_steering_config,
        specific_steering_config=specific_config,
        all_layers=all_layers,
        done_keys=done_keys,
        experiment_results=experiment_results,
        output_json=output_json,
        bias=bias,
    )

    print(f"[save] wrote → {output_json}")

    # Free memory
    del hf_model, hf_tokenizer, steering_vectors, neg_embeddings, pos_embeddings
    torch.cuda.empty_cache()


def run_steering_experiments_honesty_all_layers_wrapper(
        model_id: str,
        steering_technique: str,
        config_file: str,
        steering_mode: Optional[str] = None,
        honesty_category: str = "self_report_experimental",
        test_mode: bool = False,
        exp_name: Optional[str] = None,
):
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_save_name = model_id.replace("/", "_")

    common_steering_config = merge_generation_defaults(cfg, f"common_steering_config_{model_save_name}_{honesty_category}")
    common_steering_config["model_id"] = model_id

    common_steering_config["data_path"] = os.path.join(DATA_FOLDER, common_steering_config["data_path"])

    common_steering_config["vector_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    common_steering_config["neg_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["neg_embeddings_path"])
    common_steering_config["pos_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["pos_embeddings_path"])

    if common_steering_config.get("bias_path"):
        common_steering_config["bias_path"] = os.path.join(
            PERSONA_VECTORS_FOLDER, common_steering_config["bias_path"]
        )

    # Build specific_config dynamically (no config file lookup)
    steering_type = {"all_token_mode": "all", "response_mode": "response"}.get(steering_mode, "all")
    specific_config = {
        "steering_type": steering_type,
        "steering_direction": "positive",
        "sys_prompt": True,
        "enable_thinking": common_steering_config.get("enable_thinking", False),
        "exp_name": exp_name,
    }

    # Read coefficient sweep from config (required for non-baseline techniques)
    all_layers_cfg = cfg.get("all_layers", {})
    technique_cfg = all_layers_cfg.get(steering_technique, {})
    if "coefficients" in technique_cfg:
        specific_config["coefficients"] = technique_cfg["coefficients"]

    # Build output path dynamically
    relative = build_single_turn_output_path("honesty", steering_technique, steering_mode)
    # Insert honesty category subfolder
    relative = relative.replace("all_layers/", f"all_layers/{honesty_category}/")
    output_file_path = os.path.join(f"honesty/{model_save_name}", relative)
    output_file_path = output_file_path.replace(f"_{VECTOR_TYPE_LABEL}.json", f"_{model_save_name}_{VECTOR_TYPE_LABEL}.json")
    specific_config["output_json"] = os.path.join(RESULTS_FOLDER, VECTOR_TYPE, output_file_path)

    # Optional: nest under an `--exp_name` subdirectory to keep prior runs from
    # being overwritten when re-running with the same (model, technique, mode,
    # features) combination.
    if exp_name:
        out_dir = os.path.dirname(specific_config["output_json"])
        out_base = os.path.basename(specific_config["output_json"])
        specific_config["output_json"] = os.path.join(out_dir, exp_name, out_base)

    # Map to sequential functions
    if steering_technique == "swfc":
        steering_function = run_all_layers_swfc_sequential
    elif steering_technique == "swfc_mult_layers":
        steering_function = run_all_layers_swfc_multiple_layers_sequential

    elif steering_technique == "sttp":
        steering_function = run_all_layers_sttp_sequential
    elif steering_technique == "sttp_mult_layers":
        steering_function = run_all_layers_sttp_multiple_layers_sequential

    elif steering_technique == "stmp":
        steering_function = run_all_layers_stmp_sequential
    elif steering_technique == "stmp_mult_layers":
        steering_function = run_all_layers_stmp_multiple_layers_sequential

    elif steering_technique == "aligned_baseline" or steering_technique == "malicious_baseline":
        steering_function = run_all_layers_no_steering_sequential

    else:
        raise ValueError(f"Unknown steering technique: {steering_technique}")

    run_steering_experiments_honesty_all_layers_sequential(
        steering_function=steering_function,
        specific_config=specific_config,
        common_steering_config=common_steering_config,
        aligned_baseline=(steering_technique == "aligned_baseline"),
        test_mode=test_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sequential honesty steering over all layers for a single technique/mode."
    )
    parser.add_argument(
        "--steering_technique",
        choices=[
            "swfc", "swfc_mult_layers",
            "sttp", "sttp_mult_layers",
            "stmp", "stmp_mult_layers",
            "aligned_baseline", "malicious_baseline",
        ],
        help="Steering technique to run.",
    )
    parser.add_argument(
        "--steering_mode",
        choices=["response_mode", "all_token_mode"],
        help="Steering mode (omit for baselines).",
    )
    parser.add_argument(
        "--honesty_category", default="self_report_experimental",
        choices=["self_report_experimental"],
        help="Honesty category / dataset (default: self_report_experimental).",
    )
    parser.add_argument(
        "--model_id", default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model ID (default: meta-llama/Llama-3.3-70B-Instruct).",
    )
    parser.add_argument(
        "--test_mode", action="store_true",
        help="Run on 5 evenly-spaced layers instead of all layers (quick validation).",
    )
    parser.add_argument(
        "--exp_name", default=None,
        help=(
            "Optional experiment name. When set, results are nested under "
            "an additional `<exp_name>/` subdirectory so re-runs do not "
            "overwrite prior results."
        ),
    )

    args = parser.parse_args()

    from src.constants.constants import get_config_for_trait
    config_file = get_config_for_trait("honesty")

    run_steering_experiments_honesty_all_layers_wrapper(
        model_id=args.model_id,
        steering_technique=args.steering_technique,
        config_file=config_file,
        steering_mode=args.steering_mode,
        honesty_category=args.honesty_category,
        test_mode=args.test_mode,
        exp_name=args.exp_name,
    )
