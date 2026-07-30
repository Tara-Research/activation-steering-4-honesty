"""
Dynamic Steering Analysis Module

This module provides tools for analyzing how steering affects text generation
quality dynamically over token position, comparing experimental conditions
(malicious system prompt + corrective steering) against target conditions
(aligned system prompt + no steering).

Core metrics:
- Target distance: z-score from positive distribution mean
- Cross-entropy: naturalness under aligned model
- Divergence: L2 and cosine distance from target activations

Experimental Design:
- Target condition: aligned system prompt + no steering (gold standard)
- Experimental condition: malicious system prompt + corrective steering
- Paired comparison: same prompt, same seed, different conditions
"""

from .metrics import (
    compute_projection_stats,
    compute_target_distance,
    compute_target_distance_trajectory,
    compute_projection_trajectory,
    compute_per_token_cross_entropy,
)
from .divergence import (
    compute_divergence_trajectory,
    compute_cosine_distance_trajectory,
    compute_both_divergences,
)
from .paired_generation import (
    set_seed,
    generate_with_recording,
    create_steerer,
    generate_paired_conditions,
    compute_paired_metrics,
    compute_baseline_metrics,
)
from .constants import (
    ACTIVATION_SMOOTHING_WINDOW,
    DEFAULT_WINDOW_SIZE,
    DIVERGENCE_WINDOW_SIZE,
    EARLY_TOKEN_LIMIT,
)
from .judge_evaluation import (
    load_dynamic_results,
    evaluate_dynamic_results,
)

__all__ = [
    # Metrics
    "compute_projection_stats",
    "compute_target_distance",
    "compute_target_distance_trajectory",
    "compute_projection_trajectory",
    "compute_per_token_cross_entropy",
    # Divergence
    "compute_divergence_trajectory",
    "compute_cosine_distance_trajectory",
    "compute_both_divergences",
    # Paired generation
    "set_seed",
    "generate_with_recording",
    "create_steerer",
    "generate_paired_conditions",
    "compute_paired_metrics",
    "compute_baseline_metrics",
    # Constants
    "ACTIVATION_SMOOTHING_WINDOW",
    "DEFAULT_WINDOW_SIZE",
    "DIVERGENCE_WINDOW_SIZE",
    "EARLY_TOKEN_LIMIT",
    # Judge evaluation
    "load_dynamic_results",
    "evaluate_dynamic_results",
]
