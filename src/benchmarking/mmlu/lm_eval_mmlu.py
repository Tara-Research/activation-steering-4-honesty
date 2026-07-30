import argparse
import json
import os

import lm_eval
import torch
import yaml
from lm_eval.models.huggingface import HFLM

from src.constants import constants
from src.constants.constants import PERSONA_VECTORS_FOLDER, get_config_for_trait
from src.steering.sttp import SteeringToTargetProjection
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.stmp import SteeringToMirrorProjection
from src.utils.model_utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU benchmark with activation steering")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--honesty-category", type=str, default="self_report_experimental")
    parser.add_argument("--trait", type=str, default="compassion")
    parser.add_argument("--steering-type", type=str, default="all")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name; results saved under results/mmlu/<trait>/<model>/<exp_name>/")
    args = parser.parse_args()

    model_id = args.model_id
    trait = args.trait
    steering_type = args.steering_type
    honesty_category = args.honesty_category
    exp_name = args.exp_name

    fewshots_list = [0]#, 5]

    model_save_name = model_id.replace("/", "_")
    output_dir = os.path.join(
        constants.RESULTS_FOLDER,
        "mmlu",
        trait,
        model_save_name,
        exp_name,
    )

    config_file = get_config_for_trait(trait)
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_layers_cfg = cfg["all_layers"]
    coefficients = {
        "swfc": all_layers_cfg["swfc"]["coefficients"],
        "sttp": all_layers_cfg["sttp"]["coefficients"],
        "stmp": all_layers_cfg["stmp"]["coefficients"],
    }

    if honesty_category and trait == "honesty":
        common_steering_config = cfg[f"common_steering_config_{model_save_name}_{honesty_category}"]
    else:
        common_steering_config = cfg[f"common_steering_config_{model_save_name}"]

    common_steering_config["vector_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    common_steering_config["neg_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["neg_embeddings_path"])
    common_steering_config["pos_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["pos_embeddings_path"])

    # Load per-technique best layers from config
    mode_key = {"all": "all_token_mode", "response": "response_mode"}[steering_type]
    if honesty_category and trait == "honesty":
        best_layer_key = f"best_layer_{model_save_name}_{honesty_category}"
    else:
        best_layer_key = f"best_layer_{model_save_name}"
    best_layer_cfg = cfg[best_layer_key][mode_key]
    fixed_layer = best_layer_cfg["swfc"]["layer"]
    dist_layer = best_layer_cfg["sttp"]["layer"]
    mirror_layer = best_layer_cfg["stmp"]["layer"]

    steering_vectors = torch.load(common_steering_config["vector_path"], weights_only=False)
    neg_embeddings = torch.load(common_steering_config["neg_embeddings_path"])
    pos_embeddings = torch.load(common_steering_config["pos_embeddings_path"])

    bias = None
    if common_steering_config.get("bias_path"):
        bias_full_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["bias_path"])
        bias = torch.load(bias_full_path, weights_only=False)

    # ------------------------------------------------------------------------------------------------
    # ---------------- model -----------------
    lm = HFLM(pretrained=model_id, batch_size=16, parallelize=True, dtype="bfloat16")
    base_model = lm.model

    set_seed(constants.SEED)

    def mmlu_metrics(out):
        m = out["results"]["mmlu"]
        return {"acc,none": float(m["acc,none"]), "acc_stderr,none": float(m["acc_stderr,none"])}

    # ------------------------------------------
        
    for num_fewshots in fewshots_list:
        results = {}

        # baseline
        results_baseline_no_steering = lm_eval.simple_evaluate(
            model=lm,
            tasks=["mmlu"],
            num_fewshot=num_fewshots,
        )
        results["baseline_no_steering"] = mmlu_metrics(results_baseline_no_steering)
        print("baseline_no_steering", results["baseline_no_steering"])

        # fixed steering
        results["swfc"] = {}
        vector = steering_vectors[fixed_layer]

        for coefficient in coefficients["swfc"]:
            with SteeringWithFixedCoef(base_model, vector, coeff=coefficient, layer_idx=fixed_layer, positions=steering_type):
                results_fixed = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=["mmlu"],
                    num_fewshot=num_fewshots,
                )

            results["swfc"][str(coefficient)] = mmlu_metrics(results_fixed)
            print("swfc", coefficient, results["swfc"][str(coefficient)])

        # distribution steering
        results["sttp"] = {}
        for coefficient in coefficients["sttp"]:
            with SteeringToTargetProjection(base_model, steering_vectors=steering_vectors, coefficient=coefficient, layer_idx=dist_layer, positions=steering_type, negative_embeddings=neg_embeddings, positive_embeddings=pos_embeddings, bias=bias):
                results_dist = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=["mmlu"],
                    num_fewshot=num_fewshots,
                )

            results["sttp"][str(coefficient)] = mmlu_metrics(results_dist)
            print("sttp", coefficient, results["sttp"][str(coefficient)])

        # mirror steering
        results["stmp"] = {}

        for coefficient in coefficients["stmp"]:
            with SteeringToMirrorProjection(base_model, steering_vectors=steering_vectors, coefficient=coefficient, layer_idx=mirror_layer, positions=steering_type, negative_embeddings=neg_embeddings, positive_embeddings=pos_embeddings, bias=bias):
                results_mirror = lm_eval.simple_evaluate(
                    model=lm,
                    tasks=["mmlu"],
                    num_fewshot=num_fewshots,
                )

            results["stmp"][str(coefficient)] = mmlu_metrics(results_mirror)
            print("stmp", coefficient, results["stmp"][str(coefficient)])

    # ------------------ save ------------------

        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/mmlu_results_{steering_type}_few_shots{num_fewshots}_2.json", "w") as f:
            json.dump(results, f, indent=2)

