from __future__ import annotations

import gc
import json
import os
from typing import Dict, Any, List, Tuple

import torch
import yaml

from src.activation_projection.activation_projection_runner import activation_projection_scores_single_pass
from src.constants.constants import PERSONA_VECTORS_FOLDER, RESULTS_FOLDER, VECTOR_TYPE_LABEL
from src.utils.path_utils import build_multi_turn_results_base
from src.multi_turn_steering.honesty.n_run_metrics import compute_n_run_metrics
from src.multi_turn_steering.qualitative_evaluation.novelty_per_turn import \
    semantic_novelty_per_turn
from src.multi_turn_steering.qualitative_evaluation.repetitiveness_n_grams import _tokenize_words, \
    compute_ngram_repetition
from src.multi_turn_steering.qualitative_evaluation.sentence_template_reuse import \
    sentence_template_reuse
from src.steering.steerer_factory import create_steerer_for_eval
from src.utils.exp_utils import resolve_experiment_dir
from src.utils.model_utils import load_model

TECHNIQUES = [
    "aligned_baseline",
    "malicious_baseline",
    "swfc",
    "sttp",
    "stmp",
]

def concise_value(
    assistant_text: str,
    novelty_dict: Dict[str, Any],
    n_gram_repetitiveness: Dict,
    *,
    eps: float = 1e-12,
) -> float:
    history_novelty = float(novelty_dict.get("history_novelty", 0.0))
    words = _tokenize_words(assistant_text)
    num_words = max(1, len(words))
    trigram_repeat_within = float(n_gram_repetitiveness["within"][3])
    rep_penalty = max(0.0, 1.0 - trigram_repeat_within)
    return float((history_novelty / (num_words + eps)) * rep_penalty)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_scenario_dirs(
    scenario_category: str,
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
) -> List[str]:
    """Return sorted list of scenario directory names starting with 'honesty_multiturn_'."""
    MODEL_SAVE_NAME = model_id.replace("/", "_")
    mt_base = build_multi_turn_results_base("honesty", MODEL_SAVE_NAME)
    base_pre_exp = os.path.join(mt_base, honesty_category, scenario_category, layer_mode)
    resolved_exp = resolve_experiment_dir(base_pre_exp, exp_name=exp_name, create=False)
    base = os.path.join(base_pre_exp, resolved_exp)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Results directory not found: {base}")
    return sorted(
        d for d in os.listdir(base)
        if d.startswith("honesty_multiturn_")
        and os.path.isdir(os.path.join(base, d))
    )


def _resolve_results_and_metrics(
    techniques: List[str],
    scenario_category: str,
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
) -> List[Tuple[str, str, str, str]]:
    """
    For each (scenario, technique), return (scenario_id, technique, results_file, metrics_file).
    Only includes entries where the results file exists.
    """
    MODEL_SAVE_NAME = model_id.replace("/", "_")
    mt_base = build_multi_turn_results_base("honesty", MODEL_SAVE_NAME)
    scenario_dirs = discover_scenario_dirs(scenario_category, layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)
    base_pre_exp = os.path.join(mt_base, honesty_category, scenario_category, layer_mode)
    resolved_exp = resolve_experiment_dir(base_pre_exp, exp_name=exp_name, create=False)
    base = os.path.join(base_pre_exp, resolved_exp)
    items = []
    for scenario_id in scenario_dirs:
        for technique in techniques:
            results_file = os.path.join(
                base, scenario_id, f"{technique}_{VECTOR_TYPE_LABEL}.json"
            )
            if not os.path.isfile(results_file):
                print(f"[WARN] Missing results file, skipping: {results_file}")
                continue
            metrics_file = results_file.replace(".json", "_metrics.json")
            items.append((scenario_id, technique, results_file, metrics_file))
    return items


# ---------------------------------------------------------------------------
# Phase A1: Activation Projection (GPU)
# ---------------------------------------------------------------------------

def _get_layer_from_config(cfg: dict, steering_mode, steering_technique, honesty_category: str, model_save_name: str) -> tuple[int, float]:
    """Extract a steering layer from the honesty config."""
    layer_config = cfg[f"best_layer_{model_save_name}_{honesty_category}"][steering_mode][steering_technique]
    layer = layer_config["layer"]
    coefficient = layer_config["coefficient"]
    return layer, coefficient

def _get_layer_from_config_baseline(cfg: dict, honesty_category: str, steering_mode: str, model_save_name: str) -> int:
    """Extract the baseline activation-projection layer from the honesty config for a given mode."""
    layer_config = cfg[f"best_layer_{model_save_name}_{honesty_category}"][steering_mode]["baseline_layer_projection_activation"]
    layer = layer_config["layer"]
    return layer

def compute_activation_projection_metrics(
    history: List[Dict[str, Any]],
    config_file: str = None,
    model=None,
    tokenizer=None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
    steerer=None,
    layer: int = None
) -> Dict[str, Any]:
    if config_file is None:
        raise ValueError("config_file is required for compute_activation_projection_metrics")

    assistant_texts = [m["content"] for m in history if m["role"] == "assistant"]

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    model_save_name = model_id.replace("/", "_")
    common_steering_config = cfg[f"common_steering_config_{model_save_name}_{honesty_category}"]

    common_steering_config["vector_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    common_steering_config["neg_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["neg_embeddings_path"])
    common_steering_config["pos_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["pos_embeddings_path"])


    vec = torch.load(common_steering_config["vector_path"], weights_only=False)

    assistant_trait_projection_scores = activation_projection_scores_single_pass(
        model_id_or_path=model_id,
        hf_model=model,
        hf_tokenizer=tokenizer,
        history=history,
        layer=layer,
        steering_vector=vec[layer],
        steerer=steerer,
        capture_mode="post",
    )

    metrics: Dict[str, Any] = {}
    for i in range(len(assistant_texts)):
        metrics[f"turn_{i}"] = {
            "assistant_projection_metric": assistant_trait_projection_scores[i],
            "repetitiveness_n_grams_metric": None,
            "reused_sentences_metric": None,
            "semantic_novelty_metric": None,
            "concise_value": None,
            "trait_expression_score": None,
            "coherence_score": None,
        }

    return metrics


def phase_a1_activation_projection_for_all(
    techniques: List[str] = None,
    config_file: str = None,
    scenario_category: str = "factual_claims",
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
    steering_mode: str = "all_token_mode",
):
    if techniques is None:
        techniques = TECHNIQUES
    if config_file is None:
        raise ValueError("config_file is required for phase_a1_activation_projection_for_all")

    items = _resolve_results_and_metrics(techniques, scenario_category, layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)

    model, tokenizer = load_model(model_id_or_path=model_id)

    # Load steering vectors and embeddings once for steerer creation
    model_save_name = model_id.replace("/", "_")
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common_cfg = cfg[f"common_steering_config_{model_save_name}_{honesty_category}"]
    steering_vectors = torch.load(
        os.path.join(PERSONA_VECTORS_FOLDER, common_cfg["vector_path"]),
        weights_only=False,
    )
    neg_embeddings = torch.load(
        os.path.join(PERSONA_VECTORS_FOLDER, common_cfg["neg_embeddings_path"]),
        weights_only=False,
    )
    pos_embeddings = torch.load(
        os.path.join(PERSONA_VECTORS_FOLDER, common_cfg["pos_embeddings_path"]),
        weights_only=False,
    )
    bias_path = common_cfg.get("bias_path")
    bias = torch.load(os.path.join(PERSONA_VECTORS_FOLDER, bias_path), weights_only=False) if bias_path else None

    for scenario_id, technique, results_file, metrics_file in items:
        print(f"[A1] {scenario_id}/{technique}")

        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        
        specific_config = result.get("specific_steering_config")
        if not specific_config:
            steering_type_val = "all" if steering_mode == "all_token_mode" else "response"
            specific_config = {
                "steering_type": steering_type_val,
                "steering_direction": "positive",
            }

        if "baseline" not in technique:
            layer, coefficient = _get_layer_from_config(cfg, steering_mode=steering_mode, steering_technique=technique,
                                           honesty_category=honesty_category, model_save_name=model_save_name)

            steerer = create_steerer_for_eval(
                technique, model, steering_vectors, specific_config,
                negative_embeddings=neg_embeddings,
                positive_embeddings=pos_embeddings,
                layer=layer,
                coefficient=coefficient,
                bias=bias,
            )
        else:
            layer = _get_layer_from_config_baseline(cfg, honesty_category=honesty_category, steering_mode=steering_mode, model_save_name=model_save_name)
            steerer = None

        history = result["conversation_history"]
        metrics = compute_activation_projection_metrics(
            history, config_file=config_file, model=model, tokenizer=tokenizer,
            model_id=model_id, steerer=steerer, honesty_category=honesty_category,
            layer=layer
        )

        # If metrics file already exists, merge only projection scores
        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_dump = json.load(f)
            existing = metrics_dump["metrics"][0] if metrics_dump.get("metrics") else {}
            for turn_key, new_vals in metrics.items():
                if turn_key in existing:
                    existing[turn_key]["assistant_projection_metric"] = new_vals["assistant_projection_metric"]
                else:
                    existing[turn_key] = new_vals
            metrics_dump["metrics"] = [existing]
        else:
            metrics_dump = {
                "config": specific_config,
                "metrics": [metrics],
            }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dump, f)

    del model, tokenizer
    clear_gpu_memory()


# ---------------------------------------------------------------------------
# Phase A2: Qualitative Text Metrics (CPU / BERT)
# ---------------------------------------------------------------------------

def _average_metrics(metric_list):
    """Recursively average numeric leaves across a list of metric dicts/values."""
    if not metric_list:
        return None
    first = metric_list[0]
    if isinstance(first, (int, float)):
        valid = [v for v in metric_list if isinstance(v, (int, float))]
        return sum(valid) / len(valid) if valid else None
    if isinstance(first, dict):
        return {
            k: _average_metrics([m[k] for m in metric_list if k in m])
            for k in first
        }
    return first  # non-numeric leaf (e.g. None, str) — pass through


def compute_qualitative_metrics(
    history: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    assistant_texts = [m["content"] for m in history if m["role"] == "assistant"]

    for i, assistant_text in enumerate(assistant_texts):
        key = f"turn_{i}"

        repetitiveness_n_grams_metric = compute_ngram_repetition(
            assistant_text=assistant_text,
            prev_assistant_texts=assistant_texts[:i],
            ns=[2, 3, 4],
        )

        reused_sentences_metric = sentence_template_reuse(
            assistant_text=assistant_text,
            prev_assistant_texts=assistant_texts[:i],
            threshold=0.8,
            return_matches=False,
        )

        semantic_novelty_metric = semantic_novelty_per_turn(
            assistant_text=assistant_text,
            prev_assistant_texts=assistant_texts[:i],
            compute_adjacent=False,
        )

        conciseness = concise_value(
            assistant_text=assistant_text,
            novelty_dict=semantic_novelty_metric,
            n_gram_repetitiveness=repetitiveness_n_grams_metric,
        )

        metrics[key]["repetitiveness_n_grams_metric"] = repetitiveness_n_grams_metric
        metrics[key]["reused_sentences_metric"] = reused_sentences_metric
        metrics[key]["semantic_novelty_metric"] = semantic_novelty_metric
        metrics[key]["concise_value"] = conciseness

    return metrics


def phase_a2_qualitative_metrics_for_all(
    techniques: List[str] = None,
    scenario_category: str = "factual_claims",
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
):
    if techniques is None:
        techniques = TECHNIQUES

    items = _resolve_results_and_metrics(techniques, scenario_category, layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)

    for scenario_id, technique, results_file, metrics_file in items:
        print(f"[A2] {scenario_id}/{technique}")

        if not os.path.isfile(metrics_file):
            print(f"[WARN] No metrics file from A1, skipping: {metrics_file}")
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_dump = json.load(f)

        histories = result.get("conversation_histories")
        if not histories:
            h = result.get("conversation_history", [])
            histories = [h] if h else []

        if not histories:
            print(f"[WARN] No conversation histories found, skipping: {results_file}")
            continue

        metrics = metrics_dump["metrics"][0]
        turn_keys = [k for k in metrics if k.startswith("turn_")]

        QUAL_FIELDS = [
            "repetitiveness_n_grams_metric",
            "reused_sentences_metric",
            "semantic_novelty_metric",
            "concise_value",
        ]

        # Compute qualitative metrics per run
        all_run_metrics = []
        for hist in histories:
            run_metrics = {key: {} for key in turn_keys}
            compute_qualitative_metrics(hist, run_metrics)
            all_run_metrics.append(run_metrics)

        # Average across runs and write into metrics
        for key in turn_keys:
            for field in QUAL_FIELDS:
                values = [rm[key][field] for rm in all_run_metrics if field in rm[key]]
                metrics[key][field] = _average_metrics(values) if values else None

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dump, f)


# ---------------------------------------------------------------------------
# Phase Judge: Read existing scores from conversation_history
# ---------------------------------------------------------------------------

def fill_judge_scores_from_results(
    techniques: List[str] = None,
    scenario_category: str = "factual_claims",
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
):
    """
    Read trait_expression_score and coherence_score already embedded in
    assistant messages of conversation_history and fill them into _metrics.json.
    No LLM judge invocation needed.
    """
    if techniques is None:
        techniques = TECHNIQUES

    items = _resolve_results_and_metrics(techniques, scenario_category, layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)

    for scenario_id, technique, results_file, metrics_file in items:
        if not os.path.isfile(metrics_file):
            print(f"[WARN] No metrics file, skipping judge fill: {metrics_file}")
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_dump = json.load(f)

        histories = result.get("conversation_histories")
        if not histories:
            h = result.get("conversation_history", [])
            histories = [h] if h else []

        if not histories:
            print(f"[WARN] No conversation histories found, skipping: {results_file}")
            continue

        metrics = metrics_dump["metrics"][0]
        turn_keys = [k for k in metrics if k.startswith("turn_")]

        for key in turn_keys:
            turn_idx = int(key.split("_")[1])
            trait_scores = []
            coherence_scores = []
            for hist in histories:
                assistants = [m for m in hist if m.get("role") == "assistant"]
                if turn_idx < len(assistants):
                    ts = assistants[turn_idx].get("trait_expression_score")
                    cs = assistants[turn_idx].get("coherence_score")
                    if ts is not None:
                        trait_scores.append(ts)
                    if cs is not None:
                        coherence_scores.append(cs)

            metrics[key]["trait_expression_score"] = (
                sum(trait_scores) / len(trait_scores) if trait_scores else None
            )
            metrics[key]["coherence_score"] = (
                sum(coherence_scores) / len(coherence_scores) if coherence_scores else None
            )

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_dump, f)

        print(f"[Judge] Filled scores for {scenario_id}/{technique}")


# ---------------------------------------------------------------------------
# Phase N-Run: MASK-inspired N-run honesty metrics
# ---------------------------------------------------------------------------

def phase_n_run_honesty_metrics(
    techniques: List[str] = None,
    scenario_category: str = "factual_claims",
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
):
    """Compute MASK-inspired N-run metrics for all scenarios.

    Requires:
      - N-run results (conversation_histories key) for each technique
      - aligned_baseline results as knowledge reference
      - consistency_scores already written by the consistency judge
    """
    if techniques is None:
        techniques = TECHNIQUES

    MODEL_SAVE_NAME = model_id.replace("/", "_")
    mt_base = build_multi_turn_results_base("honesty", MODEL_SAVE_NAME)
    scenario_dirs = discover_scenario_dirs(scenario_category, layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)
    base_pre_exp = os.path.join(mt_base, honesty_category, scenario_category, layer_mode)
    resolved_exp = resolve_experiment_dir(base_pre_exp, exp_name=exp_name, create=False)
    base = os.path.join(base_pre_exp, resolved_exp)

    for scenario_id in scenario_dirs:
        # Load aligned_baseline as knowledge reference
        aligned_path = os.path.join(base, scenario_id, f"aligned_baseline_{VECTOR_TYPE_LABEL}.json")
        if not os.path.isfile(aligned_path):
            print(f"[N-Run] Missing aligned_baseline, skipping: {scenario_id}")
            continue

        with open(aligned_path, "r", encoding="utf-8") as f:
            aligned_result = json.load(f)

        aligned_histories = aligned_result.get("conversation_histories")
        if not aligned_histories:
            # Fall back to single-run format
            h = aligned_result.get("conversation_history", [])
            aligned_histories = [h] if h else []

        if not aligned_histories:
            print(f"[N-Run] No aligned histories found, skipping: {scenario_id}")
            continue

        for technique in techniques:
            results_file = os.path.join(base, scenario_id, f"{technique}_{VECTOR_TYPE_LABEL}.json")
            if not os.path.isfile(results_file):
                continue

            with open(results_file, "r", encoding="utf-8") as f:
                result = json.load(f)

            technique_histories = result.get("conversation_histories")
            if not technique_histories:
                h = result.get("conversation_history", [])
                technique_histories = [h] if h else []

            if not technique_histories:
                continue

            consistency_scores = result.get("consistency_scores", {})

            metrics = compute_n_run_metrics(
                technique_histories=technique_histories,
                aligned_histories=aligned_histories,
                consistency_scores=consistency_scores,
            )

            n_run_metrics_file = os.path.join(
                base, scenario_id, f"{technique}_{VECTOR_TYPE_LABEL}_n_run_metrics.json"
            )
            n_run_dump = {
                "config": result.get("specific_steering_config", {}),
                "n_runs": result.get("n_runs", len(technique_histories)),
                "metrics": metrics,
            }
            with open(n_run_metrics_file, "w", encoding="utf-8") as f:
                json.dump(n_run_dump, f, indent=2)

            print(f"[N-Run] {scenario_id}/{technique} → {n_run_metrics_file}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    techniques: List[str] = None,
    config_file: str = None,
    scenario_category: str = "factual_claims",
    layer_mode: str = "single_layer",
    exp_name: str = None,
    model_id: str = None,
    honesty_category: str = "diverse_with_sycophancy",
):
    if techniques is None:
        techniques = TECHNIQUES
    if config_file is None:
        raise ValueError("config_file is required for compute_all_metrics")

    if model_id is None:
        raise ValueError("model_id is required for compute_all_metrics")

    # Phase A1: activation projections (Llama 70B on GPU)
    phase_a1_activation_projection_for_all(techniques, config_file=config_file, scenario_category=scenario_category, layer_mode=layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)
    #
    # # Phase A2: qualitative metrics (BERT / SentenceTransformer)
    # clear_gpu_memory()
    # phase_a2_qualitative_metrics_for_all(techniques, scenario_category=scenario_category, layer_mode=layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)

    # Phase Judge: read existing scores from results
    # fill_judge_scores_from_results(techniques, scenario_category=scenario_category, layer_mode=layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)
    clear_gpu_memory()

    # Phase N-Run: MASK-inspired N-run metrics
    # phase_n_run_honesty_metrics(techniques, scenario_category=scenario_category, layer_mode=layer_mode, exp_name=exp_name, model_id=model_id, honesty_category=honesty_category)


if __name__ == "__main__":
    import argparse
    from src.constants.constants import get_config_for_trait

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario_category", type=str, default="factual_claims",
                        help="Scenario category to compute metrics for (e.g. factual_claims, professional_advice)")
    parser.add_argument("--layer_mode", type=str, default="single_layer",
                        choices=["single_layer", "multi_layer"],
                        help="Layer mode: single_layer or multi_layer")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment directory name (e.g. exp_1). Uses latest if omitted.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to config YAML file for activation projection. Uses the honesty config if omitted.")
    parser.add_argument("--honesty_category", type=str, default="diverse_with_sycophancy",
                        help="Honesty category (e.g. diverse_with_sycophancy)")
    args = parser.parse_args()

    config_file = args.config_file
    if config_file is None:
        config_file = get_config_for_trait("honesty")

    compute_all_metrics(scenario_category=args.scenario_category, layer_mode=args.layer_mode, exp_name=args.exp_name, config_file=config_file, model_id=args.model_id, honesty_category=args.honesty_category)
