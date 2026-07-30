"""
Dynamic Steering Analysis Experiment Runner

Runs comprehensive analysis comparing experimental (malicious + steering)
vs target (aligned + no steering) conditions across all steering methods.

Trait-agnostic: driven entirely by the YAML config file passed at runtime.
Each steering method uses its own best layer from the config's best_layer section.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
import fire
import torch
import numpy as np
import yaml
from tqdm import tqdm

from src.constants import constants
from src.utils.model_utils import load_model
from src.dynamic_analysis.paired_generation import (
    generate_with_recording,
    forward_pass_with_recording,
    compute_paired_metrics,
    compute_baseline_metrics,
    set_seed,
    create_steerer,
)
from src.dynamic_analysis.metrics import compute_projection_stats
from src.constants.constants import PERSONA_VECTORS_FOLDER, RESULTS_FOLDER, DATA_FOLDER
from src.utils.exp_utils import resolve_experiment_dir
from scipy import stats as scipy_stats


# Maps short method names (CLI) to config YAML keys in best_layer
METHOD_NAME_MAP = {
    "fixed": "swfc",
    "distribution": "sttp",
    "mirror": "stmp",
}


def _load_method_coefficients(cfg: dict, methods: list) -> Dict[str, list]:
    """Load method coefficient sweep ranges from config's all_layers section."""
    all_layers_cfg = cfg["all_layers"]
    method_coefficients = {}
    for method in methods:
        config_key = METHOD_NAME_MAP[method]
        method_coefficients[method] = all_layers_cfg[config_key]["coefficients"]
    return method_coefficients


def load_all_layers_results(
    results_dir: str,
    method: str,
    steering_mode: str,
    trait: str,
    model_save_name: str,
    vector_type_label: str,
    layer: int,
    coefficient_idx: int,
    subdir: str = "",
) -> List[Dict[str, str]]:
    """
    Load steered results from all_layers experiment JSON files.

    Args:
        results_dir: Path to the all_layers results directory
            (e.g., results/logistic_regression/compassion/.../all_layers/)
        method: Steering method config key (e.g., "swfc")
        steering_mode: "all_token" or "response"
        trait: Trait name (e.g., "compassion" or "honesty")
        model_save_name: Model name with slashes replaced by underscores
        vector_type_label: Short label (e.g., "logreg")
        layer: Layer number to load results for
        coefficient_idx: Coefficient value used in JSON key (coef_3, coef_6, ...)
        subdir: Run-specific nesting between the technique dir and filename
            (e.g. ``04_06_llama_hon``); "" for flat layout.

    Returns:
        List of dicts with {question, steered_answer} per prompt
    """
    filename = f"{trait}_{method}_{steering_mode}_mode_{model_save_name}_{vector_type_label}.json"
    filepath = os.path.join(results_dir, method, subdir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    layer_data = data["layers"][str(layer)]
    coef_key = f"coef_{coefficient_idx}"
    return layer_data[coef_key]


def load_baseline_results(
    results_dir: str,
    trait: str,
    model_save_name: str,
    vector_type_label: str,
    aligned: bool = True,
    subdir: str = "",
) -> List[Dict[str, str]]:
    """
    Load no_steering baseline results from all_layers experiment.

    Args:
        results_dir: Path to the all_layers results directory
        trait: Trait name (e.g., "compassion" or "honesty")
        model_save_name: Model name with slashes replaced by underscores
        vector_type_label: Short label (e.g., "logreg")
        aligned: True for aligned baseline, False for malicious baseline
        subdir: Run-specific nesting between the no_steering dir and filename
            (e.g. ``04_06_llama_hon``); "" for flat layout.

    Returns:
        List of dicts with {question, steered_answer} per prompt
    """
    baseline_type = "aligned" if aligned else "malicious"
    filename = f"{trait}_{baseline_type}_baseline_{model_save_name}_{vector_type_label}.json"
    filepath = os.path.join(results_dir, "no_steering", subdir, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Baseline files are flat lists
    return data


def load_scenarios_with_system_prompts(
    data_path: str,
    n_prompts: int,
) -> List[Dict[str, str]]:
    """
    Load honesty scenarios with per-scenario system prompts.

    Each scenario has its own
    system_prompt_honest and system_prompt_dishonest.

    Args:
        data_path: Path to the scenario JSON file
        n_prompts: Maximum number of scenarios to load

    Returns:
        List of dicts with {user_message, system_prompt_honest, system_prompt_dishonest}
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    return scenarios[:n_prompts]


def compute_ci_95(values: list) -> tuple:
    """
    Compute 95% confidence interval using t-distribution.

    This provides a statistically rigorous measure of uncertainty in the
    aggregate metrics, enabling proper comparison between methods.

    Args:
        values: List of values to compute CI for

    Returns:
        (lower, upper) bounds of 95% CI, or (nan, nan) if insufficient data
    """
    # Filter out non-finite values
    finite_values = [v for v in values if np.isfinite(v)]
    if len(finite_values) < 2:
        return (np.nan, np.nan)

    n = len(finite_values)
    mean = np.mean(finite_values)
    se = scipy_stats.sem(finite_values)  # Standard error of the mean

    # Use t-distribution for CI (appropriate for small samples)
    ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
    return (float(ci[0]), float(ci[1]))


@dataclass
class ExperimentConfig:
    """Configuration for dynamic steering analysis experiment."""
    # Model
    model_id_or_path: str
    model_save_name: str
    vector_path: str
    neg_embeddings_path: str
    pos_embeddings_path: str
    bias_path: str

    # System prompts
    aligned_sys_prompt: str
    malicious_sys_prompt: str

    # Prompts
    data_path: str

    # Experiment parameters
    methods: List[str]
    method_layers: Dict[str, int]
    typical_coefficients: Dict[str, float]
    method_coefficients: Dict[str, list]
    max_tokens: int = 2048
    n_prompts: int = 50
    seed_base: int = 42
    temperature: float = 0.6
    top_p: float = 0.9
    steering_type: str = "response"
    steering_direction: str = "positive"

    # Output
    exp_name: str = ""
    output_dir: str = ""


#TODO this has to be updated in order to work for honesty!!
def load_user_questions(user_questions_path: str, n_prompts: int) -> List[str]:
    """Load user prompts from file."""
    with open(user_questions_path, "r") as f:
        data = json.load(f)

    if 'user_questions_test' in data:
        data = data['user_questions_test']
    if 'scenarios' in data:
        data = data['scenarios']
    # Handle different formats
    if isinstance(data, dict):
        # Check for common keys that contain prompts
        for key in ["user_message", "prompts", "questions", "data"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError(f"Could not find prompts list in dict: {data.keys()}")

    if isinstance(data, list):
        if isinstance(data[0], str):
            prompts = data
        elif isinstance(data[0], dict):
            # Assume key is "prompt" or "question" or "content"
            for key in ["user_message", "user_question", "prompt", "question", "content", "user_prompt"]:
                if key in data[0]:
                    prompts = [d[key] for d in data]
                    break
            else:
                raise ValueError(f"Could not find prompt key in data: {data[0].keys()}")
        else:
            raise ValueError(f"Unexpected data format: {type(data[0])}")
    else:
        raise ValueError(f"Expected list, got {type(data)}")

    return prompts[:n_prompts]


def run_single_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    method: str,
    coefficient: float,
    layer: int,
    prompts: List[str],
    steering_vectors: torch.Tensor,
    neg_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    bias: Optional[torch.Tensor],
    mu_pos: float,
    sigma_pos: float,
    aligned_cache: Optional[Dict[int, Dict[str, Any]]] = None,
    loaded_steered: Optional[List[Dict[str, str]]] = None,
    scenarios: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Run experiment for a single (method, coefficient, layer) combination.

    Args:
        ... (existing args)
        aligned_cache: Optional dict mapping prompt_index -> aligned generation results.
            If provided, reuses cached aligned baselines instead of regenerating.
            This is a major optimization: avoids redundant TARGET generations.
        loaded_steered: Optional list of pre-generated steered results to use
            instead of autoregressive generation. Each entry has {question, steered_answer}.
        scenarios: Optional list of scenario dicts with per-scenario system prompts
            (honesty test scenarios).

    Returns:
        dict with prompts results and aggregate statistics
    """
    results = []
    for i, user_prompt in enumerate(tqdm(prompts, desc=f"{method} coef={coefficient} layer={layer}")):
        seed = config.seed_base + i

        try:
            # Get TARGET from cache or generate (cache is the optimization)
            if aligned_cache is not None and i in aligned_cache:
                target = aligned_cache[i]
            else:
                # Fallback: generate if not cached (shouldn't happen in normal use)
                set_seed(seed)
                target = generate_with_recording(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=(scenarios[i]["system_prompt_honest"]
                                   if scenarios is not None
                                   else config.aligned_sys_prompt),
                    user_prompt=user_prompt,
                    layer_idx=layer,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    steerer=None,
                )

            # Create steerer for EXPERIMENTAL condition
            steerer = create_steerer(
                model=model,
                method=method,
                steering_vectors=steering_vectors,
                layer_idx=layer,
                coefficient=coefficient,
                steering_direction=config.steering_direction,
                steering_type=config.steering_type,
                negative_embeddings=neg_embeddings,
                positive_embeddings=pos_embeddings,
                bias=bias,
            )

            if loaded_steered is not None:
                # Forward pass on pre-generated text with steering hooks active
                malicious_sys = (scenarios[i]["system_prompt_dishonest"]
                                 if scenarios is not None
                                 else config.malicious_sys_prompt)
                experimental = forward_pass_with_recording(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=malicious_sys,
                    user_prompt=user_prompt,
                    response_text=loaded_steered[i]["steered_answer"],
                    layer_idx=layer,
                    steerer=steerer,
                )
            else:
                # Generate EXPERIMENTAL (malicious + steering)
                set_seed(seed)
                experimental = generate_with_recording(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=(scenarios[i]["system_prompt_dishonest"]
                                   if scenarios is not None
                                   else config.malicious_sys_prompt),
                    user_prompt=user_prompt,
                    layer_idx=layer,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    steerer=steerer,
                )

            # Compute metrics (including cross-entropy under target model)
            aligned_sys_for_metrics = (scenarios[i]["system_prompt_honest"]
                                       if scenarios is not None
                                       else config.aligned_sys_prompt)
            metrics = compute_paired_metrics(
                target=target,
                experimental=experimental,
                steering_vector=steering_vectors[layer],
                mu_pos=mu_pos,
                sigma_pos=sigma_pos,
                model=model,
                tokenizer=tokenizer,
                aligned_sys_prompt=aligned_sys_for_metrics,
                user_prompt=user_prompt,
            )

            results.append({
                "user_prompt": user_prompt,
                "seed": seed,
                "target_text": target["text"],
                "experimental_text": experimental["text"],
                "metrics": metrics,
            })

        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            results.append({
                "user_prompt": user_prompt,
                "seed": seed,
                "error": str(e),
            })

    # Compute aggregate statistics
    valid_results = [r for r in results if "metrics" in r]
    if valid_results:
        # Track overflow events with diagnostic info for divergence
        divergence_overflow_events = []
        divergences = []
        for i, r in enumerate(valid_results):
            div = r["metrics"].get("mean_divergence", np.nan)
            if np.isfinite(div):
                divergences.append(div)
            else:
                # Record diagnostic info for overflow
                div_traj = r["metrics"].get("divergence_trajectory", [])
                first_inf_idx = next(
                    (j for j, v in enumerate(div_traj) if not np.isfinite(v)), -1
                )
                divergence_overflow_events.append({
                    "prompt_index": i,
                    "user_prompt": r.get("user_prompt", "")[:100],
                    "first_overflow_token": first_inf_idx,
                    "token_count_experimental": r["metrics"].get("token_count_experimental", -1),
                    "token_count_target": r["metrics"].get("token_count_target", -1),
                })

        # Track overflow events for cross-entropy (filter inf AND nan)
        cross_entropy_overflow_events = []
        cross_entropies = []
        for i, r in enumerate(valid_results):
            ppl = r["metrics"].get("mean_cross_entropy", np.nan)
            if np.isfinite(ppl):
                cross_entropies.append(ppl)
            else:
                ce_traj = r["metrics"].get("cross_entropy_trajectory", []) or []
                first_inf_idx = next(
                    (j for j, v in enumerate(ce_traj) if not np.isfinite(v)), -1
                ) if ce_traj else -1
                cross_entropy_overflow_events.append({
                    "prompt_index": i,
                    "user_prompt": r.get("user_prompt", "")[:100],
                    "first_overflow_token": first_inf_idx,
                    "mean_cross_entropy_raw": ppl,  # Keep the raw value for debugging
                })

        # Baseline cross-entropy (target text under aligned model - sanity check)
        # This should be LOW since the aligned model generated this text
        baseline_cross_entropies = []
        for r in valid_results:
            baseline_ppl = r["metrics"].get("mean_cross_entropy_baseline", np.nan)
            if np.isfinite(baseline_ppl):
                baseline_cross_entropies.append(baseline_ppl)

        # Collect cosine distances, filtering out nan/inf
        cosine_distances = [
            r["metrics"]["mean_cosine_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("mean_cosine_distance", np.nan))
        ]

        # Collect early-token metrics (for fair comparison across length differences)
        early_target_distances = [
            r["metrics"]["early_mean_target_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_target_distance", np.nan))
        ]
        early_divergences = [
            r["metrics"]["early_mean_divergence"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_divergence", np.nan))
        ]
        early_cosine_distances = [
            r["metrics"]["early_mean_cosine_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_cosine_distance", np.nan))
        ]
        early_cross_entropies = [
            r["metrics"]["early_mean_cross_entropy"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_cross_entropy", np.nan))
        ]

        # Full-sequence target distances for CI computation
        target_distances = [
            r["metrics"]["mean_target_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("mean_target_distance", np.nan))
        ]

        aggregate = {
            # Full-sequence statistics
            "mean_target_distance": float(np.mean(target_distances)) if target_distances else np.nan,
            "mean_divergence": float(np.mean(divergences)) if divergences else np.nan,
            "mean_cosine_distance": float(np.mean(cosine_distances)) if cosine_distances else np.nan,
            "mean_cross_entropy": float(np.mean(cross_entropies)) if cross_entropies else np.nan,
            "mean_cross_entropy_baseline": float(np.mean(baseline_cross_entropies)) if baseline_cross_entropies else np.nan,
            "mean_token_count": float(np.mean([
                r["metrics"]["token_count_experimental"] for r in valid_results
            ])),
            "ci_95_token_count": compute_ci_95([
                r["metrics"]["token_count_experimental"] for r in valid_results
            ]),

            # Early-token statistics (first EARLY_TOKEN_LIMIT tokens - fair comparison)
            "early_mean_target_distance": float(np.mean(early_target_distances)) if early_target_distances else np.nan,
            "early_mean_divergence": float(np.mean(early_divergences)) if early_divergences else np.nan,
            "early_mean_cosine_distance": float(np.mean(early_cosine_distances)) if early_cosine_distances else np.nan,
            "early_mean_cross_entropy": float(np.mean(early_cross_entropies)) if early_cross_entropies else np.nan,

            # 95% Confidence Intervals (for statistical rigor)
            "ci_95_target_distance": compute_ci_95(target_distances),
            "ci_95_divergence": compute_ci_95(divergences),
            "ci_95_cosine_distance": compute_ci_95(cosine_distances),
            "ci_95_cross_entropy": compute_ci_95(cross_entropies),
            "ci_95_early_target_distance": compute_ci_95(early_target_distances),
            "ci_95_early_divergence": compute_ci_95(early_divergences),
            "ci_95_early_cosine_distance": compute_ci_95(early_cosine_distances),
            "ci_95_early_cross_entropy": compute_ci_95(early_cross_entropies),

            # Metadata
            "n_valid": len(valid_results),
            "n_total": len(results),
            "n_divergence_overflow": len(divergence_overflow_events),
            "divergence_overflow_events": divergence_overflow_events,
            "n_cross_entropy_overflow": len(cross_entropy_overflow_events),
            "cross_entropy_overflow_events": cross_entropy_overflow_events,
        }

        # Warn about divergence overflow events
        if divergence_overflow_events:
            print(f"  WARNING: {len(divergence_overflow_events)} prompt(s) had numerical overflow in divergence:")
            for evt in divergence_overflow_events[:3]:  # Show first 3
                print(f"    - Prompt {evt['prompt_index']}: overflow at token {evt['first_overflow_token']}, "
                      f"exp_len={evt['token_count_experimental']}, tgt_len={evt['token_count_target']}")
            if len(divergence_overflow_events) > 3:
                print(f"    ... and {len(divergence_overflow_events) - 3} more")
            print(f"  Mean divergence computed from {len(divergences)}/{len(valid_results)} valid prompts")

        # Warn about cross-entropy overflow events
        if cross_entropy_overflow_events:
            print(f"  WARNING: {len(cross_entropy_overflow_events)} prompt(s) had numerical overflow in cross-entropy:")
            for evt in cross_entropy_overflow_events[:3]:  # Show first 3
                print(f"    - Prompt {evt['prompt_index']}: first overflow at token {evt['first_overflow_token']}")
            if len(cross_entropy_overflow_events) > 3:
                print(f"    ... and {len(cross_entropy_overflow_events) - 3} more")
            print(f"  Mean cross-entropy computed from {len(cross_entropies)}/{len(valid_results)} valid prompts")
    else:
        aggregate = {"n_valid": 0, "n_total": len(results)}

    return {
        "prompts": results,
        "aggregate": aggregate,
    }


def run_aligned_steering_experiment(
    model,
    tokenizer,
    config: ExperimentConfig,
    method: str,
    coefficient: float,
    layer: int,
    prompts: List[str],
    steering_vectors: torch.Tensor,
    neg_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    bias: Optional[torch.Tensor],
    mu_pos: float,
    sigma_pos: float,
    aligned_cache: Dict[int, Dict[str, Any]],
    scenarios: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Run experiment for aligned_sys_prompt + steering condition.

    This enables comparison: "How does steering affect behavior when
    starting from an aligned prompt vs a misaligned prompt?"

    Key differences from run_single_experiment():
    - Uses config.aligned_sys_prompt for generation (not malicious)
    - Uses cached aligned baseline for divergence/metrics (same reference)
    - Computes cross-entropy under aligned model (same reference)

    Args:
        ... (same as run_single_experiment)
        aligned_cache: Required dict mapping prompt_index -> aligned generation results.
        scenarios: Optional list of scenario dicts with per-scenario system prompts.

    Returns:
        dict with prompts results and aggregate statistics
    """
    results = []
    for i, user_prompt in enumerate(tqdm(prompts, desc=f"aligned_{method} coef={coefficient} layer={layer}")):
        seed = config.seed_base + i

        try:
            # Get TARGET from cache (required for this function)
            if i not in aligned_cache:
                print(f"Warning: prompt {i} not in aligned_cache, skipping")
                continue
            target = aligned_cache[i]

            # Create steerer for EXPERIMENTAL condition
            steerer = create_steerer(
                model=model,
                method=method,
                steering_vectors=steering_vectors,
                layer_idx=layer,
                coefficient=coefficient,
                steering_direction=config.steering_direction,
                steering_type=config.steering_type,
                negative_embeddings=neg_embeddings,
                positive_embeddings=pos_embeddings,
                bias=bias,
            )

            # Generate EXPERIMENTAL (ALIGNED + steering)
            # This is the key difference: we use aligned_sys_prompt, not malicious
            aligned_sys = (scenarios[i]["system_prompt_honest"]
                           if scenarios is not None
                           else config.aligned_sys_prompt)
            set_seed(seed)
            experimental = generate_with_recording(
                model=model,
                tokenizer=tokenizer,
                system_prompt=aligned_sys,
                user_prompt=user_prompt,
                layer_idx=layer,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                steerer=steerer,
            )

            # Compute metrics (same reference as misaligned condition)
            # This ensures fair comparison between aligned+steering and misaligned+steering
            metrics = compute_paired_metrics(
                target=target,
                experimental=experimental,
                steering_vector=steering_vectors[layer],
                mu_pos=mu_pos,
                sigma_pos=sigma_pos,
                model=model,
                tokenizer=tokenizer,
                aligned_sys_prompt=aligned_sys,
                user_prompt=user_prompt,
            )

            results.append({
                "user_prompt": user_prompt,
                "seed": seed,
                "target_text": target["text"],
                "experimental_text": experimental["text"],
                "metrics": metrics,
            })

        except Exception as e:
            print(f"Error processing prompt {i}: {e}")
            results.append({
                "user_prompt": user_prompt,
                "seed": seed,
                "error": str(e),
            })

    # Compute aggregate statistics (same logic as run_single_experiment)
    valid_results = [r for r in results if "metrics" in r]
    if valid_results:
        # Track overflow events with diagnostic info for divergence
        divergence_overflow_events = []
        divergences = []
        for i, r in enumerate(valid_results):
            div = r["metrics"].get("mean_divergence", np.nan)
            if np.isfinite(div):
                divergences.append(div)
            else:
                div_traj = r["metrics"].get("divergence_trajectory", [])
                first_inf_idx = next(
                    (j for j, v in enumerate(div_traj) if not np.isfinite(v)), -1
                )
                divergence_overflow_events.append({
                    "prompt_index": i,
                    "user_prompt": r.get("user_prompt", "")[:100],
                    "first_overflow_token": first_inf_idx,
                    "token_count_experimental": r["metrics"].get("token_count_experimental", -1),
                    "token_count_target": r["metrics"].get("token_count_target", -1),
                })

        # Track overflow events for cross-entropy
        cross_entropy_overflow_events = []
        cross_entropies = []
        for i, r in enumerate(valid_results):
            ppl = r["metrics"].get("mean_cross_entropy", np.nan)
            if np.isfinite(ppl):
                cross_entropies.append(ppl)
            else:
                ce_traj = r["metrics"].get("cross_entropy_trajectory", []) or []
                first_inf_idx = next(
                    (j for j, v in enumerate(ce_traj) if not np.isfinite(v)), -1
                ) if ce_traj else -1
                cross_entropy_overflow_events.append({
                    "prompt_index": i,
                    "user_prompt": r.get("user_prompt", "")[:100],
                    "first_overflow_token": first_inf_idx,
                    "mean_cross_entropy_raw": ppl,
                })

        baseline_cross_entropies = []
        for r in valid_results:
            baseline_ppl = r["metrics"].get("mean_cross_entropy_baseline", np.nan)
            if np.isfinite(baseline_ppl):
                baseline_cross_entropies.append(baseline_ppl)

        cosine_distances = [
            r["metrics"]["mean_cosine_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("mean_cosine_distance", np.nan))
        ]

        early_target_distances = [
            r["metrics"]["early_mean_target_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_target_distance", np.nan))
        ]
        early_divergences = [
            r["metrics"]["early_mean_divergence"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_divergence", np.nan))
        ]
        early_cosine_distances = [
            r["metrics"]["early_mean_cosine_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_cosine_distance", np.nan))
        ]
        early_cross_entropies = [
            r["metrics"]["early_mean_cross_entropy"] for r in valid_results
            if np.isfinite(r["metrics"].get("early_mean_cross_entropy", np.nan))
        ]

        target_distances = [
            r["metrics"]["mean_target_distance"] for r in valid_results
            if np.isfinite(r["metrics"].get("mean_target_distance", np.nan))
        ]

        aggregate = {
            "mean_target_distance": float(np.mean(target_distances)) if target_distances else np.nan,
            "mean_divergence": float(np.mean(divergences)) if divergences else np.nan,
            "mean_cosine_distance": float(np.mean(cosine_distances)) if cosine_distances else np.nan,
            "mean_cross_entropy": float(np.mean(cross_entropies)) if cross_entropies else np.nan,
            "mean_cross_entropy_baseline": float(np.mean(baseline_cross_entropies)) if baseline_cross_entropies else np.nan,
            "mean_token_count": float(np.mean([
                r["metrics"]["token_count_experimental"] for r in valid_results
            ])),
            "ci_95_token_count": compute_ci_95([
                r["metrics"]["token_count_experimental"] for r in valid_results
            ]),
            "early_mean_target_distance": float(np.mean(early_target_distances)) if early_target_distances else np.nan,
            "early_mean_divergence": float(np.mean(early_divergences)) if early_divergences else np.nan,
            "early_mean_cosine_distance": float(np.mean(early_cosine_distances)) if early_cosine_distances else np.nan,
            "early_mean_cross_entropy": float(np.mean(early_cross_entropies)) if early_cross_entropies else np.nan,
            "ci_95_target_distance": compute_ci_95(target_distances),
            "ci_95_divergence": compute_ci_95(divergences),
            "ci_95_cosine_distance": compute_ci_95(cosine_distances),
            "ci_95_cross_entropy": compute_ci_95(cross_entropies),
            "ci_95_early_target_distance": compute_ci_95(early_target_distances),
            "ci_95_early_divergence": compute_ci_95(early_divergences),
            "ci_95_early_cosine_distance": compute_ci_95(early_cosine_distances),
            "ci_95_early_cross_entropy": compute_ci_95(early_cross_entropies),
            "n_valid": len(valid_results),
            "n_total": len(results),
            "n_divergence_overflow": len(divergence_overflow_events),
            "divergence_overflow_events": divergence_overflow_events,
            "n_cross_entropy_overflow": len(cross_entropy_overflow_events),
            "cross_entropy_overflow_events": cross_entropy_overflow_events,
        }

        if divergence_overflow_events:
            print(f"  WARNING: {len(divergence_overflow_events)} prompt(s) had numerical overflow in divergence")
        if cross_entropy_overflow_events:
            print(f"  WARNING: {len(cross_entropy_overflow_events)} prompt(s) had numerical overflow in cross-entropy")
    else:
        aggregate = {"n_valid": 0, "n_total": len(results)}

    return {
        "prompts": results,
        "aggregate": aggregate,
    }


def run_dynamic_analysis(
    config_path: str = constants.CONFIG_HONESTY,
    model_id_or_path: str = "meta-llama/Llama-3.3-70B-Instruct",
    methods: str = "fixed,distribution,mirror",
    honesty_category: str = None,
    max_tokens: int = 2048,
    n_prompts: int = 50,
    seed_base: int = 42,
    temperature: float = 0.6,
    top_p: float = 0.9,
    exp_name: str = None,
    load_from_results: bool = True,
    results_dir: str = None,
    input_exp_name: str = None,
):
    """
    Run full dynamic analysis experiment.

    Trait-agnostic: the config file's dynamic_analysis section drives all
    trait-specific behavior (system prompts, data paths, result paths).

    Args:
        config_path: Path to YAML config (e.g. configs/config_honesty.yaml)
        model_id_or_path: Model identifier or path
        methods: Comma-separated list of steering methods
        max_tokens: Maximum tokens to generate
        n_prompts: Number of prompts to use
        seed_base: Base random seed
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        exp_name: Experiment name for resolve_experiment_dir (auto-increments if None)
        load_from_results: If True, load pre-generated responses from all_layers
            results and extract activations via a single forward pass instead of
            autoregressive generation. Dramatically faster.
        results_dir: Override for the all_layers results dir holding the technique
            subdirs (swfc/sttp/stmp/no_steering). If None, auto-derived from
            trait/model/honesty_category.
        input_exp_name: Experiment-name subdir of the steering results being loaded
            (e.g. ``04_06_llama_hon``). Distinct from the output ``exp_name``.
    """
    if not load_from_results:
        raise ValueError("load_from_results=False is no longer supported due to computational cost. Please set load_from_results=True to use pre-generated responses from all_layers experiments.")

    # ---- Load YAML config ----
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_save_name = model_id_or_path.replace("/", "_")

    # Fallback to config's dynamic_analysis.honesty_category if not passed via CLI
    dynamic_cfg_pre = cfg.get("dynamic_analysis", {})
    if honesty_category is None:
        honesty_category = dynamic_cfg_pre.get("honesty_category")

    # Common config with model-specific key (matching multi-turn pattern)
    if honesty_category:
        common_key = f"common_steering_config_{model_save_name}_{honesty_category}"
    else:
        common_key = f"common_steering_config_{model_save_name}"

    common_steering_config = cfg[common_key]

    # Parse comma-separated args (handle string, tuple/list, and single value inputs)
    if isinstance(methods, str):
        methods_list = [m.strip() for m in methods.split(",")]
    elif isinstance(methods, (list, tuple)):
        methods_list = list(methods)
    else:
        methods_list = [methods]

    # ---- System prompts ----
    # Two dataset shapes exist. Per-scenario datasets (honesty) carry a
    # system_prompt_honest/system_prompt_dishonest pair on every scenario, so no global
    # pair applies and the config declares no system_prompts_path. Single-pair datasets
    # (compassion) declare one, and every scenario uses it. Which shape is in play is
    # resolved below, once `scenarios` is loaded.
    dynamic_cfg = cfg["dynamic_analysis"]
    if "system_prompts_path" in dynamic_cfg:
        sys_prompts_path = os.path.join(DATA_FOLDER, dynamic_cfg["system_prompts_path"])
        with open(sys_prompts_path, "r") as f:
            sys_prompts = yaml.safe_load(f)
        aligned_sys_prompt = sys_prompts[dynamic_cfg["aligned_sys_prompt_key"]]
        malicious_sys_prompt = sys_prompts[dynamic_cfg["malicious_sys_prompt_key"]]
    else:
        aligned_sys_prompt = None
        malicious_sys_prompt = None

    # ---- Vector paths via PERSONA_VECTORS_FOLDER ----
    vector_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    neg_embeddings_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["neg_embeddings_path"])
    pos_embeddings_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["pos_embeddings_path"])

    bias_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["bias_path"])

    # ---- Data path via DATA_FOLDER ----
    data_path = os.path.join(DATA_FOLDER, common_steering_config["data_path"])

    # ---- Per-method layer and typical coefficient mapping from best_layer ----
    if honesty_category:
        best_layer_root = cfg.get(f"best_layer_{model_save_name}_{honesty_category}", {})
        if not best_layer_root:
            best_layer_root = cfg.get(f"best_layer_{model_save_name}", {})
    else:
        best_layer_root = cfg.get(f"best_layer_{model_save_name}", {})

    steering_type = dynamic_cfg.get("steering_type", "response")
    mode_key = "all_token_mode" if steering_type == "all" else "response_mode"
    best_layer_cfg = best_layer_root.get(mode_key, {})
    method_layers = {}
    typical_coefficients = {}
    for method in methods_list:
        config_key = METHOD_NAME_MAP.get(method, method)
        method_layers[method] = best_layer_cfg[config_key]["layer"]
        typical_coefficients[method] = best_layer_cfg[config_key]["coefficient"]

    method_coefficients = _load_method_coefficients(cfg, methods_list)

    # ---- Result paths with RESULTS_FOLDER + resolve_experiment_dir ----
    trait = dynamic_cfg["trait"]
    base_dir = os.path.join(RESULTS_FOLDER, constants.VECTOR_TYPE, trait, model_save_name, "dynamic_analysis")
    resolved_exp_name = resolve_experiment_dir(base_dir, exp_name=exp_name, create=True)
    output_dir = os.path.join(base_dir, resolved_exp_name)

    # ---- Build ExperimentConfig ----
    config = ExperimentConfig(
        model_id_or_path=model_id_or_path,
        model_save_name=model_save_name,
        vector_path=vector_path,
        neg_embeddings_path=neg_embeddings_path,
        pos_embeddings_path=pos_embeddings_path,
        bias_path=bias_path,
        aligned_sys_prompt=aligned_sys_prompt,
        malicious_sys_prompt=malicious_sys_prompt,
        data_path=data_path,
        methods=methods_list,
        method_layers=method_layers,
        typical_coefficients=typical_coefficients,
        method_coefficients=method_coefficients,
        max_tokens=max_tokens,
        n_prompts=n_prompts,
        seed_base=seed_base,
        temperature=temperature,
        top_p=top_p,
        steering_type=dynamic_cfg.get("steering_type", "response"),
        steering_direction=dynamic_cfg.get("steering_direction", "positive"),
        exp_name=resolved_exp_name,
        output_dir=output_dir,
    )

    print(f"Configuration:")
    print(f"  Model: {config.model_id_or_path}")
    print(f"  Trait: {trait}")
    print(f"  Methods: {config.methods}")
    print(f"  Method layers: {config.method_layers}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  N prompts: {config.n_prompts}")
    print(f"  Output: {config.output_dir}")
    print(f"  Load from results: {load_from_results}")

    # ---- Set up load_from_results paths and data ----
    scenarios = None  # Will hold per-scenario system prompts for honesty
    all_layers_dir = None
    input_subdir = input_exp_name or ""
    vector_type_label = constants.VECTOR_TYPE_LABEL
    steering_mode = "all_token" if steering_type == "all" else "response"

    if load_from_results:
        # Build path to all_layers results directory
        if results_dir is not None:
            all_layers_dir = results_dir
        else:
            all_layers_dir = os.path.join(RESULTS_FOLDER, constants.VECTOR_TYPE, trait, model_save_name, "all_layers")
            if honesty_category:
                all_layers_dir = os.path.join(all_layers_dir, honesty_category)
        print(f"  All-layers results dir: {all_layers_dir}")
        print(f"  Run subdir: {input_subdir or '(flat)'}")

        # For honesty, load per-scenario system prompts
        if trait == "honesty" and honesty_category:
            scenarios = load_scenarios_with_system_prompts(data_path, config.n_prompts)
            print(f"  Loaded {len(scenarios)} scenarios with per-scenario system prompts")

    # Without per-scenario prompts, the global pair is the only source; refuse to run
    # with neither rather than generating under an empty system prompt.
    if scenarios is None and (aligned_sys_prompt is None or malicious_sys_prompt is None):
        raise ValueError(
            f"No system prompts available for trait '{trait}': this run has no per-scenario "
            "prompts, so the config's dynamic_analysis section must set 'system_prompts_path' "
            "plus 'aligned_sys_prompt_key' and 'malicious_sys_prompt_key'."
        )

    # Load prompts
    print("Loading prompts...")
    if scenarios is not None:
        # Use scenario user_messages as prompts
        prompts = [s["user_message"] for s in scenarios]
    else:
        prompts = load_user_questions(config.data_path, config.n_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(config.model_id_or_path)
    model.eval()

    # Load steering vectors and embeddings
    print("Loading steering vectors and embeddings...")
    steering_vectors = torch.load(config.vector_path, weights_only=False)
    neg_embeddings = torch.load(config.neg_embeddings_path, weights_only=False)
    pos_embeddings = torch.load(config.pos_embeddings_path, weights_only=False)

    bias = torch.load(config.bias_path, weights_only=False)

    # Convert to float32 for numerical stability in KDE computations
    # (float16/bfloat16 cause singular covariance matrices in gaussian_kde)
    if steering_vectors.dtype in (torch.bfloat16, torch.float16):
        steering_vectors = steering_vectors.to(torch.float32)
    if neg_embeddings.dtype in (torch.bfloat16, torch.float16):
        neg_embeddings = neg_embeddings.to(torch.float32)
    if pos_embeddings.dtype in (torch.bfloat16, torch.float16):
        pos_embeddings = pos_embeddings.to(torch.float32)

    # Group methods by their best layer for efficient baseline caching
    layer_to_methods = defaultdict(list)
    for method in config.methods:
        layer_to_methods[config.method_layers[method]].append(method)
    unique_layers = sorted(layer_to_methods.keys())

    # Compute reference statistics PER LAYER from positive embeddings
    # Each layer has different mu_pos, sigma_pos due to different activation distributions
    print("Computing per-layer reference statistics...")
    reference_stats = {}
    for layer in unique_layers:
        pos_stats = compute_projection_stats(
            embeddings=pos_embeddings[layer],
            steering_vector=steering_vectors[layer],
        )
        reference_stats[layer] = {
            "mu_pos": pos_stats["mu"],
            "sigma_pos": pos_stats["sigma"],
        }
        print(f"  Layer {layer}: mu_pos={pos_stats['mu']:.3f}, sigma_pos={pos_stats['sigma']:.3f}")

    # Run experiments
    all_results = {
        "experiment_id": f"dynamic_{config.max_tokens}_{int(time.time())}",
        "config": asdict(config),
        "reference_stats": reference_stats,  # Dict keyed by layer
        "results": {},
    }

    # Count total experiments
    total_experiments = 0
    for method in config.methods:
        coeffs = config.method_coefficients[method]
        total_experiments += len(coeffs)
    print(f"\nRunning {total_experiments} experiments across {len(unique_layers)} unique layer(s)...")

    # ==========================================================================
    # LAYER-FIRST LOOP with method deduplication
    #
    # Structure:
    #   For each unique layer (from best_layer config):
    #     1. Generate aligned baselines -> CACHE activations for reuse
    #     2. Generate malicious baselines (for reference plots)
    #     3. Run all (method, coefficient) experiments for methods at this layer
    #     4. Run aligned+steering experiments for methods at this layer
    #     5. Clear cache before next layer
    #
    # This eliminates redundant TARGET generations when multiple methods
    # share the same best layer.
    # ==========================================================================

    for layer in unique_layers:
        methods_at_layer = layer_to_methods[layer]

        print(f"\n{'='*60}")
        print(f"LAYER {layer}: Methods {methods_at_layer}")
        print(f"{'='*60}")

        # ---------------------------------------------------------------------
        # STEP 1: Generate ALIGNED baselines and CACHE for reuse
        # The cache stores full generation results including activations
        # ---------------------------------------------------------------------
        if load_from_results:
            print(f"\n  [1/4] Loading aligned baselines from results + forward pass...")
            loaded_aligned = load_baseline_results(
                results_dir=all_layers_dir,
                trait=trait,
                model_save_name=model_save_name,
                vector_type_label=vector_type_label,
                aligned=True,
                subdir=input_subdir,
            )
        else:
            print(f"\n  [1/4] Generating aligned baselines (caching for reuse)...")

        aligned_cache = {}  # prompt_index -> full generation result
        baseline_results = []

        for i, prompt in enumerate(tqdm(prompts[:config.n_prompts], desc=f"aligned_layer_{layer}")):
            set_seed(config.seed_base + i)
            try:
                if load_from_results:
                    # Validate prompt matches loaded results
                    loaded_question = loaded_aligned[i]["question"]
                    if loaded_question.strip() != prompt.strip():
                        print(f"    WARNING: Prompt mismatch at index {i}:")
                        print(f"      Expected: {prompt[:80]}...")
                        print(f"      Got:      {loaded_question[:80]}...")

                    # Get per-scenario system prompt for honesty, or use config default
                    aligned_sys = (scenarios[i]["system_prompt_honest"]
                                   if scenarios is not None
                                   else config.aligned_sys_prompt)

                    aligned_gen = forward_pass_with_recording(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=aligned_sys,
                        user_prompt=prompt,
                        response_text=loaded_aligned[i]["steered_answer"],
                        layer_idx=layer,
                        steerer=None,
                    )
                else:
                    aligned_gen = generate_with_recording(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=(scenarios[i]["system_prompt_honest"]
                                       if scenarios is not None
                                       else config.aligned_sys_prompt),
                        user_prompt=prompt,
                        layer_idx=layer,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        steerer=None,
                    )

                # CACHE the full result for reuse in steering experiments
                aligned_cache[i] = aligned_gen

                # Compute baseline metrics
                aligned_sys_for_metrics = (scenarios[i]["system_prompt_honest"]
                                           if scenarios is not None
                                           else config.aligned_sys_prompt)
                baseline_metrics = compute_baseline_metrics(
                    baseline=aligned_gen,
                    steering_vector=steering_vectors[layer],
                    mu_pos=reference_stats[layer]["mu_pos"],
                    sigma_pos=reference_stats[layer]["sigma_pos"],
                    model=model,
                    tokenizer=tokenizer,
                    aligned_sys_prompt=aligned_sys_for_metrics,
                    user_prompt=prompt,
                )

                baseline_results.append({
                    "user_prompt": prompt,
                    "baseline": {
                        "text": aligned_gen["text"],
                        "token_count": len(aligned_gen["token_ids"]),
                    },
                    "metrics": baseline_metrics,
                })
            except Exception as e:
                print(f"    Error on aligned prompt {i}: {e}")
                baseline_results.append({
                    "user_prompt": prompt,
                    "error": str(e),
                })

        print(f"    Cached {len(aligned_cache)} aligned generations for reuse")

        # ---------------------------------------------------------------------
        # STEP 2: Generate MALICIOUS baselines (for reference plots)
        # These use cached aligned baselines for comparison
        # ---------------------------------------------------------------------
        if load_from_results:
            print(f"\n  [2/4] Loading malicious baselines from results + forward pass...")
            loaded_malicious = load_baseline_results(
                results_dir=all_layers_dir,
                trait=trait,
                model_save_name=model_save_name,
                vector_type_label=vector_type_label,
                aligned=False,
                subdir=input_subdir,
            )
        else:
            print(f"\n  [2/4] Generating malicious baselines...")

        malicious_baseline_results = []

        for i, prompt in enumerate(tqdm(prompts[:config.n_prompts], desc=f"malicious_layer_{layer}")):
            set_seed(config.seed_base + i)
            try:
                if load_from_results:
                    malicious_sys = (scenarios[i]["system_prompt_dishonest"]
                                     if scenarios is not None
                                     else config.malicious_sys_prompt)
                    malicious_gen = forward_pass_with_recording(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=malicious_sys,
                        user_prompt=prompt,
                        response_text=loaded_malicious[i]["steered_answer"],
                        layer_idx=layer,
                        steerer=None,
                    )
                else:
                    malicious_gen = generate_with_recording(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=(scenarios[i]["system_prompt_dishonest"]
                                       if scenarios is not None
                                       else config.malicious_sys_prompt),
                        user_prompt=prompt,
                        layer_idx=layer,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        steerer=None,
                    )

                # Compare malicious to aligned using CACHED aligned baseline
                aligned_sys_for_metrics = (scenarios[i]["system_prompt_honest"]
                                           if scenarios is not None
                                           else config.aligned_sys_prompt)
                if i in aligned_cache:
                    malicious_metrics = compute_paired_metrics(
                        experimental=malicious_gen,
                        target=aligned_cache[i],  # FROM CACHE
                        steering_vector=steering_vectors[layer],
                        mu_pos=reference_stats[layer]["mu_pos"],
                        sigma_pos=reference_stats[layer]["sigma_pos"],
                        model=model,
                        tokenizer=tokenizer,
                        aligned_sys_prompt=aligned_sys_for_metrics,
                        user_prompt=prompt,
                    )
                else:
                    # Fallback if aligned failed
                    malicious_metrics = compute_baseline_metrics(
                        baseline=malicious_gen,
                        steering_vector=steering_vectors[layer],
                        mu_pos=reference_stats[layer]["mu_pos"],
                        sigma_pos=reference_stats[layer]["sigma_pos"],
                        model=model,
                        tokenizer=tokenizer,
                        aligned_sys_prompt=aligned_sys_for_metrics,
                        user_prompt=prompt,
                    )

                malicious_baseline_results.append({
                    "user_prompt": prompt,
                    "malicious_baseline": {
                        "text": malicious_gen["text"],
                        "token_count": len(malicious_gen["token_ids"]),
                    },
                    "metrics": malicious_metrics,
                })
            except Exception as e:
                print(f"    Error on malicious prompt {i}: {e}")
                malicious_baseline_results.append({
                    "user_prompt": prompt,
                    "error": str(e),
                })

        # Compute aggregate stats for baseline
        valid_baseline = [r for r in baseline_results if "metrics" in r]
        if valid_baseline:
            baseline_cross_entropies = []
            baseline_target_distances = []
            baseline_early_target_distances = []
            baseline_early_cross_entropies = []
            for r in valid_baseline:
                ppl = r["metrics"].get("mean_cross_entropy", np.nan)
                if np.isfinite(ppl):
                    baseline_cross_entropies.append(ppl)
                td = r["metrics"].get("mean_target_distance", np.nan)
                if np.isfinite(td):
                    baseline_target_distances.append(td)
                etd = r["metrics"].get("early_mean_target_distance", np.nan)
                if np.isfinite(etd):
                    baseline_early_target_distances.append(etd)
                eppl = r["metrics"].get("early_mean_cross_entropy", np.nan)
                if np.isfinite(eppl):
                    baseline_early_cross_entropies.append(eppl)

            baseline_aggregate = {
                "n_valid": len(valid_baseline),
                "n_total": len(baseline_results),
                # Full-sequence stats
                "mean_target_distance": float(np.mean(baseline_target_distances)) if baseline_target_distances else np.nan,
                "mean_projection": float(np.mean([
                    r["metrics"]["mean_projection"] for r in valid_baseline
                ])),
                "mean_divergence": 0.0,  # Perfect by definition
                "mean_cosine_distance": 0.0,  # Perfect by definition
                "mean_cross_entropy": float(np.mean(baseline_cross_entropies)) if baseline_cross_entropies else np.nan,
                "mean_token_count": float(np.mean([
                    r["baseline"]["token_count"] for r in valid_baseline
                ])),
                "ci_95_token_count": compute_ci_95([
                    r["baseline"]["token_count"] for r in valid_baseline
                ]),
                # Early-token stats (first EARLY_TOKEN_LIMIT tokens)
                "early_mean_target_distance": float(np.mean(baseline_early_target_distances)) if baseline_early_target_distances else np.nan,
                "early_mean_divergence": 0.0,  # Perfect by definition
                "early_mean_cosine_distance": 0.0,  # Perfect by definition
                "early_mean_cross_entropy": float(np.mean(baseline_early_cross_entropies)) if baseline_early_cross_entropies else np.nan,
                # 95% Confidence Intervals
                "ci_95_target_distance": compute_ci_95(baseline_target_distances),
                "ci_95_cross_entropy": compute_ci_95(baseline_cross_entropies),
                "ci_95_early_target_distance": compute_ci_95(baseline_early_target_distances),
                "ci_95_early_cross_entropy": compute_ci_95(baseline_early_cross_entropies),
            }
        else:
            baseline_aggregate = {"n_valid": 0, "n_total": len(baseline_results)}

        # Store baseline results
        baseline_key = f"baseline_layer_{layer}"
        all_results["results"][baseline_key] = {
            "prompts": baseline_results,
            "aggregate": baseline_aggregate,
        }

        print(f"\n  Aligned baseline:")
        print(f"    Valid: {baseline_aggregate.get('n_valid', 0)}/{baseline_aggregate.get('n_total', 0)}")
        if baseline_aggregate.get("n_valid", 0) > 0:
            print(f"    Mean target distance: {baseline_aggregate.get('mean_target_distance', 'N/A'):.3f}")
            print(f"    Mean divergence: {baseline_aggregate.get('mean_divergence', 'N/A'):.3f}")
            ppl = baseline_aggregate.get('mean_cross_entropy', np.nan)
            ppl_str = f"{ppl:.2f}" if np.isfinite(ppl) else "N/A"
            print(f"    Mean cross-entropy: {ppl_str}")
            print(f"    Mean token count: {baseline_aggregate.get('mean_token_count', 'N/A'):.1f}")

        # Compute aggregate stats for malicious_baseline
        valid_malicious = [r for r in malicious_baseline_results if "metrics" in r]
        if valid_malicious:
            # Collect metrics, filtering out nan/inf
            malicious_cross_entropies = []
            malicious_divergences = []
            malicious_cosine_distances = []
            malicious_target_distances = []
            malicious_early_target_distances = []
            malicious_early_divergences = []
            malicious_early_cosine_distances = []
            malicious_early_cross_entropies = []
            for r in valid_malicious:
                ppl = r["metrics"].get("mean_cross_entropy", np.nan)
                if np.isfinite(ppl):
                    malicious_cross_entropies.append(ppl)
                div = r["metrics"].get("mean_divergence", np.nan)
                if np.isfinite(div):
                    malicious_divergences.append(div)
                cos = r["metrics"].get("mean_cosine_distance", np.nan)
                if np.isfinite(cos):
                    malicious_cosine_distances.append(cos)
                td = r["metrics"].get("mean_target_distance", np.nan)
                if np.isfinite(td):
                    malicious_target_distances.append(td)
                # Early metrics
                etd = r["metrics"].get("early_mean_target_distance", np.nan)
                if np.isfinite(etd):
                    malicious_early_target_distances.append(etd)
                ediv = r["metrics"].get("early_mean_divergence", np.nan)
                if np.isfinite(ediv):
                    malicious_early_divergences.append(ediv)
                ecos = r["metrics"].get("early_mean_cosine_distance", np.nan)
                if np.isfinite(ecos):
                    malicious_early_cosine_distances.append(ecos)
                eppl = r["metrics"].get("early_mean_cross_entropy", np.nan)
                if np.isfinite(eppl):
                    malicious_early_cross_entropies.append(eppl)

            malicious_aggregate = {
                "n_valid": len(valid_malicious),
                "n_total": len(malicious_baseline_results),
                # Full-sequence stats
                "mean_target_distance": float(np.mean(malicious_target_distances)) if malicious_target_distances else np.nan,
                "mean_divergence": float(np.mean(malicious_divergences)) if malicious_divergences else np.nan,
                "mean_cosine_distance": float(np.mean(malicious_cosine_distances)) if malicious_cosine_distances else np.nan,
                "mean_cross_entropy": float(np.mean(malicious_cross_entropies)) if malicious_cross_entropies else np.nan,
                "mean_token_count": float(np.mean([
                    r["malicious_baseline"]["token_count"] for r in valid_malicious
                ])),
                "ci_95_token_count": compute_ci_95([
                    r["malicious_baseline"]["token_count"] for r in valid_malicious
                ]),
                # Early-token stats (first EARLY_TOKEN_LIMIT tokens)
                "early_mean_target_distance": float(np.mean(malicious_early_target_distances)) if malicious_early_target_distances else np.nan,
                "early_mean_divergence": float(np.mean(malicious_early_divergences)) if malicious_early_divergences else np.nan,
                "early_mean_cosine_distance": float(np.mean(malicious_early_cosine_distances)) if malicious_early_cosine_distances else np.nan,
                "early_mean_cross_entropy": float(np.mean(malicious_early_cross_entropies)) if malicious_early_cross_entropies else np.nan,
                # 95% Confidence Intervals
                "ci_95_target_distance": compute_ci_95(malicious_target_distances),
                "ci_95_divergence": compute_ci_95(malicious_divergences),
                "ci_95_cosine_distance": compute_ci_95(malicious_cosine_distances),
                "ci_95_cross_entropy": compute_ci_95(malicious_cross_entropies),
                "ci_95_early_target_distance": compute_ci_95(malicious_early_target_distances),
                "ci_95_early_divergence": compute_ci_95(malicious_early_divergences),
                "ci_95_early_cosine_distance": compute_ci_95(malicious_early_cosine_distances),
                "ci_95_early_cross_entropy": compute_ci_95(malicious_early_cross_entropies),
            }
        else:
            malicious_aggregate = {"n_valid": 0, "n_total": len(malicious_baseline_results)}

        # Store malicious_baseline results
        malicious_key = f"malicious_baseline_layer_{layer}"
        all_results["results"][malicious_key] = {
            "prompts": malicious_baseline_results,
            "aggregate": malicious_aggregate,
        }

        print(f"\n    Malicious baseline summary:")
        print(f"      Valid: {malicious_aggregate.get('n_valid', 0)}/{malicious_aggregate.get('n_total', 0)}")
        if malicious_aggregate.get("n_valid", 0) > 0:
            print(f"      Mean target distance: {malicious_aggregate.get('mean_target_distance', 'N/A'):.3f}")
            div = malicious_aggregate.get('mean_divergence', np.nan)
            div_str = f"{div:.3f}" if np.isfinite(div) else "N/A"
            print(f"      Mean divergence: {div_str}")
            ppl = malicious_aggregate.get('mean_cross_entropy', np.nan)
            ppl_str = f"{ppl:.2f}" if np.isfinite(ppl) else "N/A"
            print(f"      Mean cross-entropy: {ppl_str}")

        # ---------------------------------------------------------------------
        # STEP 3: Run steering experiments (reusing cached aligned baselines)
        # Only run methods whose best layer matches this layer
        # ---------------------------------------------------------------------
        if load_from_results:
            print(f"\n  [3/4] Loading steered results + forward pass (reusing {len(aligned_cache)} cached baselines)...")
        else:
            print(f"\n  [3/4] Running misaligned + steering experiments (reusing {len(aligned_cache)} cached baselines)...")

        for method in methods_at_layer:
            config_method_key = METHOD_NAME_MAP[method]
            coeffs = config.method_coefficients[method]
            for coef_idx, coef in enumerate(coeffs):
                key = f"{method}_coef_{coef}_layer_{layer}"
                print(f"\n    Running: {key}")

                loaded_steered = None
                if load_from_results:
                    loaded_steered = load_all_layers_results(
                        results_dir=all_layers_dir,
                        method=config_method_key,
                        steering_mode=steering_mode,
                        trait=trait,
                        model_save_name=model_save_name,
                        vector_type_label=vector_type_label,
                        layer=layer,
                        coefficient_idx=coef,
                        subdir=input_subdir,
                    )

                result = run_single_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    method=method,
                    coefficient=coef,
                    layer=layer,
                    prompts=prompts,
                    steering_vectors=steering_vectors,
                    neg_embeddings=neg_embeddings,
                    pos_embeddings=pos_embeddings,
                    bias=bias,
                    mu_pos=reference_stats[layer]["mu_pos"],
                    sigma_pos=reference_stats[layer]["sigma_pos"],
                    aligned_cache=aligned_cache,
                    loaded_steered=loaded_steered,
                    scenarios=scenarios,
                )

                all_results["results"][key] = result

                # Print aggregate stats
                agg = result["aggregate"]
                print(f"      Valid: {agg.get('n_valid', 0)}/{agg.get('n_total', 0)}")
                if agg.get("n_valid", 0) > 0:
                    print(f"      Mean target distance: {agg.get('mean_target_distance', 'N/A'):.3f}")
                    print(f"      Mean divergence: {agg.get('mean_divergence', 'N/A'):.3f}")
                    ppl = agg.get('mean_cross_entropy', np.nan)
                    ppl_str = f"{ppl:.2f}" if np.isfinite(ppl) else "N/A"
                    print(f"      Mean cross-entropy: {ppl_str}")

                # Save intermediate results
                output_path = os.path.join(config.output_dir, f"dynamic_analysis_{all_results['experiment_id']}.json")
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

        # ---------------------------------------------------------------------
        # STEP 4: Run ALIGNED + STEERING experiments
        # Same structure as Step 3, but with aligned_sys_prompt instead of malicious
        # This enables two-panel comparison: aligned+steering vs misaligned+steering
        # ---------------------------------------------------------------------
        print(f"\n  [4/4] Running aligned + steering experiments...")

        for method in methods_at_layer:
            coeffs = config.method_coefficients[method]
            for coef in coeffs:
                key = f"aligned_{method}_coef_{coef}_layer_{layer}"
                print(f"\n    Running: {key}")

                result = run_aligned_steering_experiment(
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    method=method,
                    coefficient=coef,
                    layer=layer,
                    prompts=prompts,
                    steering_vectors=steering_vectors,
                    neg_embeddings=neg_embeddings,
                    bias=bias,
                    pos_embeddings=pos_embeddings,
                    mu_pos=reference_stats[layer]["mu_pos"],
                    sigma_pos=reference_stats[layer]["sigma_pos"],
                    aligned_cache=aligned_cache,
                    scenarios=scenarios,
                )

                all_results["results"][key] = result

                # Print aggregate stats
                agg = result["aggregate"]
                print(f"      Valid: {agg.get('n_valid', 0)}/{agg.get('n_total', 0)}")
                if agg.get("n_valid", 0) > 0:
                    print(f"      Mean target distance: {agg.get('mean_target_distance', 'N/A'):.3f}")
                    print(f"      Mean divergence: {agg.get('mean_divergence', 'N/A'):.3f}")
                    ppl = agg.get('mean_cross_entropy', np.nan)
                    ppl_str = f"{ppl:.2f}" if np.isfinite(ppl) else "N/A"
                    print(f"      Mean cross-entropy: {ppl_str}")

                # Save intermediate results
                output_path = os.path.join(config.output_dir, f"dynamic_analysis_{all_results['experiment_id']}.json")
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

        # ---------------------------------------------------------------------
        # STEP 5: Clear cache before next layer (free memory)
        # Estimate: 50 prompts x ~500 tokens x 4096 dim x 4 bytes ~ 400 MB
        # ---------------------------------------------------------------------
        cache_size_mb = len(aligned_cache) * 500 * 4096 * 4 / (1024 * 1024)
        print(f"\n  Clearing aligned cache for layer {layer} (freeing ~{cache_size_mb:.0f} MB)")
        del aligned_cache
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    fire.Fire(run_dynamic_analysis)
