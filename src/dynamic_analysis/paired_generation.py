"""
Paired generation infrastructure for dynamic steering analysis.

Generates paired conditions:
- Target: aligned system prompt + no steering
- Experimental: malicious system prompt + corrective steering

Uses identical random seeds for paired comparison.
"""

import random
from typing import Dict, Any, List, Optional, Literal, Tuple
import numpy as np
import torch

from .constants import EARLY_TOKEN_LIMIT, TRUNCATED_CE_EXCLUDE_TOKENS

from src.activation_projection.activation_recorder import ActivationRecorder
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.sttp import SteeringToTargetProjection
from src.steering.stmp import SteeringToMirrorProjection


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_with_recording(
    model: torch.nn.Module,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    layer_idx: int,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    steerer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate response while recording per-token activations.

    Args:
        model: Language model
        tokenizer: Tokenizer
        system_prompt: System prompt to use
        user_prompt: User's question/request
        layer_idx: Layer index to record activations from
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        steerer: Optional steering context manager

    Returns:
        dict with:
        - text: generated response text
        - token_ids: [T] list of token IDs
        - tokens: [T] list of token strings
        - activations: [T, H] per-token activations
        - prompt_length: number of prompt tokens
    """
    # Configure tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    prompt_length = input_ids.shape[1]

    # Set up activation recorder
    recorder = ActivationRecorder(
        model=model,
        layer_idx=layer_idx,
        to_cpu=True,
    )

    # Generate with optional steering
    # IMPORTANT: Hook registration order matters!
    # Steerer must be registered BEFORE recorder so that:
    # 1. Steerer hook runs first and modifies activations
    # 2. Recorder hook runs second and captures STEERED activations
    # If order is reversed, recorder sees unsteered activations.
    model.eval()
    if steerer is not None:
        with steerer:
            with recorder:
                with torch.no_grad():
                    output = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        do_sample=(temperature > 0),
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
    else:
        with recorder:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

    # Extract response tokens only
    response_ids = output[0, prompt_length:].tolist()
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)

    # Stack recorded activations
    if recorder.response_activations:
        activations = torch.cat(recorder.response_activations, dim=0)  # [T, H]
    else:
        activations = torch.zeros(0, model.config.hidden_size)

    # Align lengths (activations might be off by 1)
    T_ids = len(response_ids)
    T_acts = activations.shape[0]
    if T_acts != T_ids:
        min_len = min(T_ids, T_acts)
        response_ids = response_ids[:min_len]
        response_tokens = response_tokens[:min_len]
        activations = activations[:min_len]

    # Check if generation was truncated (hit max_tokens limit)
    # When truncated, cross-entropy at the end is artificially high because
    # the model was cut off mid-generation rather than choosing to end.
    is_truncated = len(response_ids) >= max_tokens - 5  # Within 5 of limit

    return {
        "text": response_text,
        "token_ids": response_ids,
        "tokens": response_tokens,
        "activations": activations,
        "prompt_length": prompt_length,
        "is_truncated": is_truncated,
    }


def forward_pass_with_recording(
    model: torch.nn.Module,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    layer_idx: int,
    steerer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run a single forward pass on pre-generated text to extract per-token activations.

    Instead of autoregressive generation (N sequential decode steps), this does
    one forward pass through the full sequence. Dramatically faster when the
    response text already exists (e.g., loaded from all_layers results).

    Args:
        model: Language model
        tokenizer: Tokenizer
        system_prompt: System prompt to use
        user_prompt: User's question/request
        response_text: Pre-generated response text to analyze
        layer_idx: Layer index to record activations from
        steerer: Optional steering context manager (hooks will modify activations)

    Returns:
        dict with:
        - text: the response text (same as input)
        - token_ids: [T] list of response token IDs
        - tokens: [T] list of response token strings
        - activations: [T, H] per-token activations for response portion
        - prompt_length: number of prompt tokens
        - is_truncated: always False (loaded text is complete)
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build prompt-only (system + user + generation prompt) to determine prompt_length
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_only = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_encoded = tokenizer(prompt_only, return_tensors="pt")
    prompt_length = prompt_encoded["input_ids"].shape[1]

    # Build full conversation (system + user + assistant response)
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response_text},
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        enable_thinking=False,
    )
    full_encoded = tokenizer(full_text, return_tensors="pt")
    input_ids = full_encoded["input_ids"].to(model.device)
    attention_mask = full_encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Set up activation recorder with response_slice
    recorder = ActivationRecorder(
        model=model,
        layer_idx=layer_idx,
        to_cpu=True,
    )
    recorder.response_slice = slice(prompt_length, None)

    # Set response_start_idx on steerer for "response" mode steering
    if steerer is not None:
        steerer.response_start_idx = prompt_length

    # Run forward pass with optional steering
    # Hook registration order: steerer BEFORE recorder so recorder captures steered activations
    model.eval()
    if steerer is not None:
        with steerer:
            with recorder:
                with torch.no_grad():
                    model(input_ids, attention_mask=attention_mask)
    else:
        with recorder:
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)

    # Extract response-portion activations from recorder
    if recorder.response_activations:
        # response_activations[0] is [1, T_response, H] from the single forward pass
        activations = recorder.response_activations[0].squeeze(0)  # [T_response, H]
    else:
        activations = torch.zeros(0, model.config.hidden_size)

    # Extract response token IDs and strings
    response_ids = input_ids[0, prompt_length:].tolist()
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)

    # Align lengths if needed
    T_ids = len(response_ids)
    T_acts = activations.shape[0]
    if T_acts != T_ids:
        min_len = min(T_ids, T_acts)
        response_ids = response_ids[:min_len]
        response_tokens = response_tokens[:min_len]
        activations = activations[:min_len]

    return {
        "text": response_text,
        "token_ids": response_ids,
        "tokens": response_tokens,
        "activations": activations,
        "prompt_length": prompt_length,
        "is_truncated": False,
    }


def create_steerer(
    model: torch.nn.Module,
    method: Literal["fixed", "distribution", "scaled", "mirror"],
    steering_vectors: torch.Tensor,
    layer_idx: int,
    coefficient: float = 1.0,
    steering_direction: str = "positive",
    steering_type: str = "response",
    negative_embeddings: Optional[torch.Tensor] = None,
    positive_embeddings: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> Any:
    """
    Create a steerer context manager based on method.

    Args:
        model: Language model
        method: Steering method name
        steering_vectors: [L, H] steering vectors per layer
        layer_idx: Layer to apply steering
        coefficient: Steering strength
        steering_direction: "positive" or "negative"
        steering_type: "response" or "all"
        negative_embeddings: [L, N, H] for distribution/mirror steering
        positive_embeddings: [L, N, H] for distribution/mirror steering

    Returns:
        Steerer context manager
    """
    if method == "fixed":
        vector = steering_vectors[layer_idx]
        return SteeringWithFixedCoef(
            model=model,
            steering_vector=vector,
            coeff=coefficient,
            layer_idx=layer_idx,
            positions=steering_type,
        )

    elif method == "distribution":
        if negative_embeddings is None or positive_embeddings is None:
            raise ValueError("Distribution steering requires embeddings")
        return SteeringToTargetProjection(
            model=model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            bias=bias,
            layer_idx=layer_idx,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
        )

    elif method == "mirror":
        if negative_embeddings is None or positive_embeddings is None:
            raise ValueError("Mirror steering requires embeddings")
        return SteeringToMirrorProjection(
            model=model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            bias=bias,
            layer_idx=layer_idx,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
        )

    else:
        raise ValueError(f"Unknown steering method: {method}")


def generate_paired_conditions(
    model: torch.nn.Module,
    tokenizer,
    user_prompt: str,
    aligned_sys_prompt: str,
    malicious_sys_prompt: str,
    steering_config: Dict[str, Any],
    layer_idx: int,
    seed: int,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate both target and experimental conditions for paired comparison.

    DEPRECATED: This function regenerates TARGET for every call, which is
    inefficient when running multiple (method, coefficient) experiments on
    the same prompts. The main experiment runner now caches aligned baselines
    and only generates EXPERIMENTAL conditions. Use generate_with_recording()
    directly with a cached baseline for better performance.

    Kept for backwards compatibility and standalone testing.

    Args:
        model: Language model
        tokenizer: Tokenizer
        user_prompt: User's question/request
        aligned_sys_prompt: System prompt for target condition
        malicious_sys_prompt: System prompt for experimental condition
        steering_config: Configuration for steering (method, vectors, etc.)
        layer_idx: Layer to record activations from and apply steering to
        seed: Random seed for reproducibility
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        target: dict with target condition results
        experimental: dict with experimental condition results
    """
    # 1. Generate TARGET (aligned + no steering)
    set_seed(seed)
    target = generate_with_recording(
        model=model,
        tokenizer=tokenizer,
        system_prompt=aligned_sys_prompt,
        user_prompt=user_prompt,
        layer_idx=layer_idx,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        steerer=None,
    )

    # 2. Generate EXPERIMENTAL (malicious + steering)
    set_seed(seed)  # Reset seed for paired comparison

    # Create steerer
    steerer = create_steerer(
        model=model,
        method=steering_config["method"],
        steering_vectors=steering_config["steering_vectors"],
        layer_idx=layer_idx,
        coefficient=steering_config.get("coefficient", 1.0),
        steering_direction=steering_config.get("steering_direction", "positive"),
        steering_type=steering_config.get("steering_type", "response"),
        negative_embeddings=steering_config.get("negative_embeddings"),
        positive_embeddings=steering_config.get("positive_embeddings"),
    )

    experimental = generate_with_recording(
        model=model,
        tokenizer=tokenizer,
        system_prompt=malicious_sys_prompt,
        user_prompt=user_prompt,
        layer_idx=layer_idx,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        steerer=steerer,
    )

    return target, experimental


def compute_paired_metrics(
    target: Dict[str, Any],
    experimental: Dict[str, Any],
    steering_vector: torch.Tensor,
    mu_pos: float,
    sigma_pos: float,
    model: Optional[torch.nn.Module] = None,
    tokenizer = None,
    aligned_sys_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute all dynamic metrics for a paired comparison.

    Args:
        target: Results from target condition
        experimental: Results from experimental condition
        steering_vector: [H] steering direction
        mu_pos: Mean projection of positive training examples
        sigma_pos: Std of positive training example projections
        model: Optional model for cross-entropy computation
        tokenizer: Optional tokenizer for cross-entropy computation
        aligned_sys_prompt: System prompt for cross-entropy context
        user_prompt: User prompt for cross-entropy context

    Returns:
        dict with all computed metrics
    """
    from .metrics import (
        compute_target_distance_trajectory,
        compute_projection_trajectory,
        compute_per_token_cross_entropy,
    )
    from .divergence import compute_both_divergences

    # Get activations
    exp_acts = experimental["activations"]
    tgt_acts = target["activations"]

    # 1. Target distance trajectory (z-scores)
    target_distance = compute_target_distance_trajectory(
        activations=exp_acts,
        steering_vector=steering_vector,
        mu_pos=mu_pos,
        sigma_pos=sigma_pos,
    )

    # 2. Projection trajectories
    exp_projections = compute_projection_trajectory(exp_acts, steering_vector)
    tgt_projections = compute_projection_trajectory(tgt_acts, steering_vector)

    # 3. Divergence from target activations (both L2 and cosine)
    # Activations are smoothed by default (window=10) before computing distances
    divergence, cosine_distance = compute_both_divergences(exp_acts, tgt_acts)

    # 4. Cross-entropy of experimental text under target model (with aligned prompt)
    # This measures: "Would the aligned model generate this text?"
    # Cross-entropy = -log P(token). Lower = more natural to aligned model.
    cross_entropy_trajectory = None
    mean_cross_entropy = np.nan
    # 4b. BASELINE: Cross-entropy of TARGET text under aligned model (sanity check)
    # This should be LOW since the aligned model generated this text
    baseline_cross_entropy_trajectory = None
    mean_cross_entropy_baseline = np.nan

    if model is not None and tokenizer is not None and aligned_sys_prompt is not None:
        try:
            # Find where assistant response starts (compute once, reuse)
            prompt_messages = [
                {"role": "system", "content": aligned_sys_prompt},
                {"role": "user", "content": user_prompt or ""},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_encoded = tokenizer(prompt_text, return_tensors="pt")
            assistant_start = prompt_encoded["input_ids"].shape[1]

            # --- EXPERIMENTAL CROSS-ENTROPY ---
            # Build the experimental response as the aligned model would see it
            # We evaluate: -log P(experimental_text | aligned_system_prompt, user_prompt)
            exp_messages = [
                {"role": "system", "content": aligned_sys_prompt},
                {"role": "user", "content": user_prompt or ""},
                {"role": "assistant", "content": experimental["text"]},
            ]
            exp_full_text = tokenizer.apply_chat_template(
                exp_messages, tokenize=False, enable_thinking=False
            )
            exp_encoded = tokenizer(exp_full_text, return_tensors="pt")

            # If experimental generation was truncated (hit max_tokens), exclude last N tokens
            # The model was cut off mid-generation, so those CE values are artificially high
            exp_exclude = TRUNCATED_CE_EXCLUDE_TOKENS if experimental.get("is_truncated", False) else 0

            exp_ce = compute_per_token_cross_entropy(
                model=model,
                input_ids=exp_encoded["input_ids"],
                response_start_idx=assistant_start,
                exclude_last_n=exp_exclude,
            )
            cross_entropy_trajectory = exp_ce.tolist()
            mean_cross_entropy = float(np.mean(exp_ce))

            # --- BASELINE CROSS-ENTROPY (TARGET TEXT) ---
            # Evaluate target text under aligned model - should be LOW
            tgt_messages = [
                {"role": "system", "content": aligned_sys_prompt},
                {"role": "user", "content": user_prompt or ""},
                {"role": "assistant", "content": target["text"]},
            ]
            tgt_full_text = tokenizer.apply_chat_template(
                tgt_messages, tokenize=False, enable_thinking=False
            )
            tgt_encoded = tokenizer(tgt_full_text, return_tensors="pt")

            # If target generation was truncated, exclude last N tokens
            tgt_exclude = TRUNCATED_CE_EXCLUDE_TOKENS if target.get("is_truncated", False) else 0

            tgt_ce = compute_per_token_cross_entropy(
                model=model,
                input_ids=tgt_encoded["input_ids"],
                response_start_idx=assistant_start,
                exclude_last_n=tgt_exclude,
            )
            baseline_cross_entropy_trajectory = tgt_ce.tolist()
            mean_cross_entropy_baseline = float(np.mean(tgt_ce))

        except Exception as e:
            print(f"Warning: Cross-entropy computation failed: {e}")
            cross_entropy_trajectory = None
            mean_cross_entropy = np.nan
            baseline_cross_entropy_trajectory = None
            mean_cross_entropy_baseline = np.nan

    # Compute early-token metrics for fair comparison across different response lengths
    # (aligned produces ~450 tokens, malicious/steered produce ~45 tokens)
    early_limit = min(EARLY_TOKEN_LIMIT, len(target_distance))
    early_div_limit = min(EARLY_TOKEN_LIMIT, len(divergence))
    early_cos_limit = min(EARLY_TOKEN_LIMIT, len(cosine_distance))

    # Aggregate metrics
    metrics = {
        # Trajectories (as lists for JSON serialization)
        "target_distance_trajectory": target_distance.tolist(),
        "divergence_trajectory": divergence.tolist(),  # L2 distance
        "cosine_distance_trajectory": cosine_distance.tolist(),  # Angular distance
        "experimental_projection_trajectory": exp_projections.tolist(),
        "target_projection_trajectory": tgt_projections.tolist(),
        "cross_entropy_trajectory": cross_entropy_trajectory,
        "baseline_cross_entropy_trajectory": baseline_cross_entropy_trajectory,

        # Aggregate statistics (full sequence)
        "mean_target_distance": float(np.mean(target_distance)),
        "mean_divergence": float(np.mean(divergence)),  # L2
        "mean_cosine_distance": float(np.mean(cosine_distance)),  # Angular
        "mean_cross_entropy": mean_cross_entropy,
        "mean_cross_entropy_baseline": mean_cross_entropy_baseline,

        # Early-token aggregate statistics (first EARLY_TOKEN_LIMIT tokens)
        # These enable fair comparison across conditions with different response lengths
        "early_mean_target_distance": float(np.mean(target_distance[:early_limit])),
        "early_mean_divergence": float(np.mean(divergence[:early_div_limit])),
        "early_mean_cosine_distance": float(np.mean(cosine_distance[:early_cos_limit])),

        "token_count_experimental": len(experimental["token_ids"]),
        "token_count_target": len(target["token_ids"]),
    }

    # Add early cross-entropy if cross-entropy was computed
    if cross_entropy_trajectory is not None:
        ce_early_limit = min(EARLY_TOKEN_LIMIT, len(cross_entropy_trajectory))
        metrics["early_mean_cross_entropy"] = float(np.mean(cross_entropy_trajectory[:ce_early_limit]))
    else:
        metrics["early_mean_cross_entropy"] = np.nan

    return metrics


def compute_baseline_metrics(
    baseline: Dict[str, Any],
    steering_vector: torch.Tensor,
    mu_pos: float,
    sigma_pos: float,
    model: Optional[torch.nn.Module] = None,
    tokenizer = None,
    aligned_sys_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for baseline condition (aligned + no steering).

    This serves as the "gold standard" - all steered methods should
    try to match these metrics. The baseline represents what the model
    naturally produces with the aligned system prompt.

    Args:
        baseline: Results from generate_with_recording for aligned + no steering
        steering_vector: [H] steering direction (will be normalized)
        mu_pos: Mean projection of positive training examples
        sigma_pos: Std of positive training example projections
        model: Optional model for cross-entropy computation
        tokenizer: Optional tokenizer for cross-entropy computation
        aligned_sys_prompt: System prompt for cross-entropy context
        user_prompt: User prompt for cross-entropy context

    Returns:
        dict with baseline metrics (trajectories and aggregates)
    """
    from .metrics import (
        compute_projection_trajectory,
        compute_target_distance_trajectory,
        compute_per_token_cross_entropy,
    )

    acts = baseline["activations"]

    # Projection trajectory
    projections = compute_projection_trajectory(acts, steering_vector)

    # Target distance (signed z-score) - use shared function to avoid duplication
    z_scores = compute_target_distance_trajectory(
        activations=acts,
        steering_vector=steering_vector,
        mu_pos=mu_pos,
        sigma_pos=sigma_pos,
    )

    # Divergence is 0 (baseline vs itself)
    divergence = np.zeros(len(projections))

    # Cross-entropy of baseline text under aligned model (self-evaluation)
    # This serves as reference - steered methods should aim for similar cross-entropy
    cross_entropy_trajectory = None
    mean_cross_entropy = np.nan

    if model is not None and tokenizer is not None and aligned_sys_prompt is not None:
        try:
            # Build baseline text as aligned model would see it
            messages = [
                {"role": "system", "content": aligned_sys_prompt},
                {"role": "user", "content": user_prompt or ""},
                {"role": "assistant", "content": baseline["text"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
            )
            encoded = tokenizer(full_text, return_tensors="pt")

            # Find where assistant response starts
            prompt_messages = [
                {"role": "system", "content": aligned_sys_prompt},
                {"role": "user", "content": user_prompt or ""},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompt_encoded = tokenizer(prompt_text, return_tensors="pt")
            assistant_start = prompt_encoded["input_ids"].shape[1]

            # If baseline generation was truncated (hit max_tokens), exclude last N tokens
            exclude_n = TRUNCATED_CE_EXCLUDE_TOKENS if baseline.get("is_truncated", False) else 0

            ce = compute_per_token_cross_entropy(
                model=model,
                input_ids=encoded["input_ids"],
                response_start_idx=assistant_start,
                exclude_last_n=exclude_n,
            )
            cross_entropy_trajectory = ce.tolist()
            mean_cross_entropy = float(np.mean(ce))

        except Exception as e:
            print(f"Warning: Baseline cross-entropy computation failed: {e}")
            cross_entropy_trajectory = None
            mean_cross_entropy = np.nan

    # Compute early-token metrics for fair comparison across different response lengths
    early_limit = min(EARLY_TOKEN_LIMIT, len(z_scores))

    # Build metrics dict
    metrics = {
        # Trajectories
        "projection_trajectory": projections.tolist(),
        "target_distance_trajectory": z_scores.tolist(),
        "divergence_trajectory": divergence.tolist(),
        "cosine_distance_trajectory": [0.0] * len(projections),  # Zero vs itself
        "cross_entropy_trajectory": cross_entropy_trajectory,

        # Aggregates (full sequence)
        "mean_target_distance": float(np.mean(z_scores)),
        "mean_projection": float(np.mean(projections)),
        "mean_divergence": 0.0,  # Perfect by definition
        "mean_cosine_distance": 0.0,  # Perfect by definition
        "mean_cross_entropy": mean_cross_entropy,

        # Early-token aggregates (first EARLY_TOKEN_LIMIT tokens)
        "early_mean_target_distance": float(np.mean(z_scores[:early_limit])),
        "early_mean_projection": float(np.mean(projections[:early_limit])),
        "early_mean_divergence": 0.0,  # Perfect by definition
        "early_mean_cosine_distance": 0.0,  # Perfect by definition

        "token_count": len(baseline["token_ids"]),
    }

    # Add early cross-entropy if cross-entropy was computed
    if cross_entropy_trajectory is not None:
        ce_early_limit = min(EARLY_TOKEN_LIMIT, len(cross_entropy_trajectory))
        metrics["early_mean_cross_entropy"] = float(np.mean(cross_entropy_trajectory[:ce_early_limit]))
    else:
        metrics["early_mean_cross_entropy"] = np.nan

    return metrics
