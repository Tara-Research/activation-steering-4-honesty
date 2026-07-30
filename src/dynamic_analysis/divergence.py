"""
Divergence analysis for comparing experimental vs target activations.

Computes per-token activation distance between:
- Experimental: malicious system prompt + corrective steering
- Target: aligned system prompt + no steering (gold standard)

Two distance metrics are provided:
- L2 (Euclidean): Measures absolute difference in activation space
- Cosine distance: Measures angular difference (direction), ignoring magnitude

By default, activations are smoothed before computing distances to reduce
noise from individual token variations.
"""

from typing import Tuple, Union
import numpy as np
import torch

from .constants import ACTIVATION_SMOOTHING_WINDOW


def _smooth_activations(
    activations: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Apply causal moving average smoothing to activations.

    Uses a causal window (only past tokens) to avoid information leakage.

    Args:
        activations: [T, H] activation matrix
        window_size: Number of tokens to average over

    Returns:
        smoothed: [T, H] smoothed activations
    """
    if window_size <= 1:
        return activations

    T, H = activations.shape
    smoothed = np.zeros_like(activations)

    # Causal cumulative sum for efficient computation
    cumsum = np.cumsum(activations, axis=0)

    for t in range(T):
        start = max(0, t - window_size + 1)
        if start == 0:
            smoothed[t] = cumsum[t] / (t + 1)
        else:
            smoothed[t] = (cumsum[t] - cumsum[start - 1]) / window_size

    return smoothed


def compute_divergence_trajectory(
    experimental_activations: Union[torch.Tensor, np.ndarray],
    target_activations: Union[torch.Tensor, np.ndarray],
    smoothing_window: int = ACTIVATION_SMOOTHING_WINDOW,
) -> np.ndarray:
    """
    Compute per-token L2 distance between experimental and target activations.

    For same prompt with same seed:
      - Target: aligned system prompt, no steering
      - Experimental: malicious system prompt, corrective steering

    Args:
        experimental_activations: [T, H] from malicious + steering
        target_activations: [T, H] from aligned + no steering (SAME PROMPT!)
        smoothing_window: Smooth activations over this many tokens before
            computing distance. Default=10 reduces noise. Set to 1 to disable.

    Returns:
        divergence: [T] L2 distance at each token position
            - Low divergence: Steering successfully matched target behavior
            - High divergence: Steering failed to correct malicious influence
    """
    # Convert to numpy
    if isinstance(experimental_activations, torch.Tensor):
        experimental_activations = experimental_activations.detach().cpu().float().numpy()
    if isinstance(target_activations, torch.Tensor):
        target_activations = target_activations.detach().cpu().float().numpy()

    # Handle length mismatch (generations may differ in length)
    T_exp = experimental_activations.shape[0]
    T_tgt = target_activations.shape[0]
    T = min(T_exp, T_tgt)

    exp = experimental_activations[:T]
    tgt = target_activations[:T]

    # Smooth activations before computing distance (reduces noise)
    if smoothing_window > 1:
        exp = _smooth_activations(exp, smoothing_window)
        tgt = _smooth_activations(tgt, smoothing_window)

    # Compute per-token L2 distance
    delta = exp - tgt  # [T, H]
    divergence = np.linalg.norm(delta, axis=1)  # [T]

    # Check for numerical issues and warn (but don't silently fix)
    n_inf = np.sum(np.isinf(divergence))
    n_nan = np.sum(np.isnan(divergence))
    if n_inf > 0 or n_nan > 0:
        problematic_idx = np.where(~np.isfinite(divergence))[0]
        first_idx = problematic_idx[0] if len(problematic_idx) > 0 else -1

        import warnings
        warnings.warn(
            f"Numerical overflow in divergence computation: "
            f"{n_inf} inf, {n_nan} nan out of {len(divergence)} tokens. "
            f"First occurrence at token {first_idx}. "
            f"This may indicate: (1) model refusal/early stopping, "
            f"(2) extreme steering correction, or (3) activation explosion. "
            f"Check the experimental vs target text lengths and content.",
            RuntimeWarning
        )

    return divergence


def compute_cosine_distance_trajectory(
    experimental_activations: Union[torch.Tensor, np.ndarray],
    target_activations: Union[torch.Tensor, np.ndarray],
    smoothing_window: int = ACTIVATION_SMOOTHING_WINDOW,
) -> np.ndarray:
    """
    Compute per-token cosine distance between experimental and target activations.

    Cosine distance measures angular difference rather than magnitude difference.
    This is useful when you care about "direction" of activations (what concepts
    are being represented) rather than their absolute values.

    Args:
        experimental_activations: [T, H] from malicious + steering
        target_activations: [T, H] from aligned + no steering
        smoothing_window: Smooth activations over this many tokens before
            computing distance. Default=10 reduces noise. Set to 1 to disable.

    Returns:
        cosine_distance: [T] values in [0, 2]
            - 0: activations point in identical direction (perfect match)
            - 1: activations are orthogonal (unrelated)
            - 2: activations point in opposite directions (inverted)

    Interpretation:
        Unlike L2 which can be ~150+ due to magnitude differences,
        cosine distance is bounded and more interpretable:
        - cos_dist < 0.1: Very similar direction
        - cos_dist ~ 0.5: Moderately different
        - cos_dist > 1.0: Mostly opposite
    """
    if isinstance(experimental_activations, torch.Tensor):
        experimental_activations = experimental_activations.detach().cpu().float().numpy()
    if isinstance(target_activations, torch.Tensor):
        target_activations = target_activations.detach().cpu().float().numpy()

    T = min(experimental_activations.shape[0], target_activations.shape[0])
    exp = experimental_activations[:T]
    tgt = target_activations[:T]

    # Smooth activations before computing distance (reduces noise)
    if smoothing_window > 1:
        exp = _smooth_activations(exp, smoothing_window)
        tgt = _smooth_activations(tgt, smoothing_window)

    # Normalize each activation vector
    exp_norms = np.linalg.norm(exp, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(tgt, axis=1, keepdims=True)

    exp_normalized = exp / (exp_norms + 1e-10)
    tgt_normalized = tgt / (tgt_norms + 1e-10)

    # Cosine similarity: dot product of normalized vectors
    cosine_similarity = np.sum(exp_normalized * tgt_normalized, axis=1)  # [T]

    # Convert to distance: 0 = identical, 2 = opposite
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


def compute_both_divergences(
    experimental_activations: Union[torch.Tensor, np.ndarray],
    target_activations: Union[torch.Tensor, np.ndarray],
    smoothing_window: int = ACTIVATION_SMOOTHING_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both L2 and cosine distance in one pass (more efficient).

    Args:
        experimental_activations: [T, H] from malicious + steering
        target_activations: [T, H] from aligned + no steering
        smoothing_window: Smooth activations over this many tokens before
            computing distance. Default=10 reduces noise. Set to 1 to disable.

    Returns:
        l2_divergence: [T] L2 (Euclidean) distance
        cosine_distance: [T] Cosine distance (angular)
    """
    if isinstance(experimental_activations, torch.Tensor):
        experimental_activations = experimental_activations.detach().cpu().float().numpy()
    if isinstance(target_activations, torch.Tensor):
        target_activations = target_activations.detach().cpu().float().numpy()

    T = min(experimental_activations.shape[0], target_activations.shape[0])
    exp = experimental_activations[:T]
    tgt = target_activations[:T]

    # Smooth activations before computing distance (reduces noise)
    if smoothing_window > 1:
        exp = _smooth_activations(exp, smoothing_window)
        tgt = _smooth_activations(tgt, smoothing_window)

    # L2 distance
    delta = exp - tgt
    l2_divergence = np.linalg.norm(delta, axis=1)

    # Cosine distance
    exp_norms = np.linalg.norm(exp, axis=1, keepdims=True)
    tgt_norms = np.linalg.norm(tgt, axis=1, keepdims=True)
    exp_normalized = exp / (exp_norms + 1e-10)
    tgt_normalized = tgt / (tgt_norms + 1e-10)
    cosine_similarity = np.sum(exp_normalized * tgt_normalized, axis=1)
    cosine_distance = 1 - cosine_similarity

    return l2_divergence, cosine_distance
