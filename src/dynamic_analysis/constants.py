"""
Constants for dynamic steering analysis.

Single source of truth for coefficients and other configuration values
used across both experiments and plotting.
"""

# =============================================================================
# EARLY-TOKEN ANALYSIS
# =============================================================================
# Early-token analysis window for fair comparison across different response lengths.
#
# SCIENTIFIC RATIONALE:
# The aligned baseline produces ~450 tokens while malicious/steered produce ~45 tokens
# (a 10x difference). When averaging trajectories of different lengths using nanmean,
# positions beyond the shortest trajectory only have data from longer sequences.
# This creates selection bias where late-sequence statistics only reflect long responses.
#
# By analyzing only the first 50 tokens, we ensure fair comparison across all conditions.
# All conditions (aligned, malicious, steered) produce at least 45 tokens, so 50 tokens
# captures the core behavior while ensuring most prompts contribute data.
EARLY_TOKEN_LIMIT = 50

# =============================================================================
# TRUNCATION HANDLING
# =============================================================================
# When a generation hits max_tokens, the model was cut off mid-generation.
# The last few tokens' cross-entropy is artificially high because the model
# didn't "choose" to end there - it was forcibly stopped.
#
# We exclude these tokens from cross-entropy calculations for truncated generations.
TRUNCATED_CE_EXCLUDE_TOKENS = 5

# =============================================================================
# ACTIVATION SMOOTHING
# =============================================================================
# Activation smoothing (applied BEFORE computing L2/cosine divergence)
# This reduces noise from individual token variations in raw activations
ACTIVATION_SMOOTHING_WINDOW = 16

# Plotting smoothing (applied AFTER metric computation, for visual clarity)
# Note: L2/cosine already have pre-smoothed activations, so no extra smoothing needed
DEFAULT_WINDOW_SIZE = 5         # For most metrics (target_distance, projection, etc.)
DIVERGENCE_WINDOW_SIZE = 1      # No extra smoothing - activations already pre-smoothed
