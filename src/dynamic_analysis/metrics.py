"""
Core metrics for dynamic steering analysis.

Provides functions to compute per-token metrics comparing experimental
(malicious + steering) vs target (aligned + no steering) conditions.
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F


def compute_projection_stats(
    embeddings: torch.Tensor,
    steering_vector: torch.Tensor,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute projection statistics for a set of embeddings.

    Args:
        embeddings: [N, H] activations from training examples
        steering_vector: [H] steering direction (will be normalized)
        eps: small value for numerical stability

    Returns:
        dict with keys: mu, sigma, min, max
    """
    if embeddings.dim() == 3:
        # [L, N, H] -> take first layer or raise error
        raise ValueError("embeddings should be [N, H], not [L, N, H]. Select a layer first.")

    # Normalize steering vector
    v = steering_vector.detach().cpu().float().numpy()
    v_norm = v / (np.linalg.norm(v) + eps)

    # Compute projections
    emb = embeddings.detach().cpu().float().numpy()
    projections = emb @ v_norm  # [N]

    return {
        "mu": float(np.mean(projections)),
        "sigma": float(np.std(projections)),
        "min": float(np.min(projections)),
        "max": float(np.max(projections)),
        "projections": projections,
    }


def compute_target_distance(
    activation: Union[torch.Tensor, np.ndarray],
    steering_vector: Union[torch.Tensor, np.ndarray],
    mu_pos: float,
    sigma_pos: float,
    eps: float = 1e-8,
) -> float:
    """
    Compute distance from target distribution for a single activation.

    The distance is measured as z-score: how many standard deviations
    the activation's projection is from the positive (target) distribution mean.

    Args:
        activation: [H] hidden state
        steering_vector: [H] steering direction (will be normalized)
        mu_pos: mean projection of positive training examples
        sigma_pos: std of positive training example projections
        eps: small value for numerical stability

    Returns:
        z-score from target distribution (lower = closer to target)
    """
    # Convert to numpy
    if isinstance(activation, torch.Tensor):
        activation = activation.detach().cpu().float().numpy()
    if isinstance(steering_vector, torch.Tensor):
        steering_vector = steering_vector.detach().cpu().float().numpy()

    # Normalize steering vector
    v_norm = steering_vector / (np.linalg.norm(steering_vector) + eps)

    # Project activation onto steering vector
    projection = float(np.dot(activation, v_norm))

    # Compute signed z-score from target (positive) distribution
    # Negative = below target mean, Positive = above target mean, Zero = at target
    z_score = (projection - mu_pos) / (sigma_pos + eps)

    return z_score


def compute_projection_trajectory(
    activations: Union[torch.Tensor, np.ndarray],
    steering_vector: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute raw projection values for each token.

    Args:
        activations: [T, H] per-token hidden states
        steering_vector: [H] steering direction (will be normalized)
        eps: small value for numerical stability

    Returns:
        projections: [T] array of scalar projection values
    """
    # Convert to numpy
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().float().numpy()
    if isinstance(steering_vector, torch.Tensor):
        steering_vector = steering_vector.detach().cpu().float().numpy()

    # Normalize steering vector
    v_norm = steering_vector / (np.linalg.norm(steering_vector) + eps)

    # Project all tokens: [T, H] @ [H] -> [T]
    projections = activations @ v_norm

    return projections


def compute_target_distance_trajectory(
    activations: Union[torch.Tensor, np.ndarray],
    steering_vector: Union[torch.Tensor, np.ndarray],
    mu_pos: float,
    sigma_pos: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute per-token signed z-score from target distribution.

    For each token t:
      1. Project activation onto steering vector: p_t = dot(h_t, v)
      2. Compute signed z-score from target: z_t = (p_t - μ_pos) / σ_pos

    Args:
        activations: [T, H] per-token hidden states
        steering_vector: [H] steering direction (will be normalized)
        mu_pos: mean projection of positive training examples
        sigma_pos: std of positive training example projections
        eps: small value for numerical stability

    Returns:
        z_scores: [T] array of signed z-scores
            - z < 0: Activation is BELOW target mean (undercorrection)
            - z = 0: Activation is at target mean (perfect)
            - z > 0: Activation is ABOVE target mean (overcorrection)
    """
    # Reuse compute_projection_trajectory to avoid duplication
    projections = compute_projection_trajectory(activations, steering_vector, eps)

    # Compute signed z-scores from target (positive) distribution
    # Negative = below target, Positive = above target, Zero = at target
    z_scores = (projections - mu_pos) / (sigma_pos + eps)

    return z_scores


def compute_per_token_cross_entropy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    response_start_idx: int = 0,
    max_value: float = 20.0,
    exclude_last_n: int = 0,
) -> np.ndarray:
    """
    Compute per-token cross-entropy (log-perplexity) for a sequence.

    Unlike perplexity (which can overflow to inf), cross-entropy stays finite
    and is more suitable for computing statistics like mean and std.

    Cross-entropy is measured in nats (natural log units).
    CE = -log(p(token)) where p is the model's predicted probability.

    Interpretation:
        CE = 0    → model assigns probability 1.0 (perfect prediction)
        CE = 1    → model assigns probability ~0.37
        CE = 2    → model assigns probability ~0.14
        CE = 5    → model assigns probability ~0.007
        CE = 10   → model assigns probability ~0.00005

    Why we might exclude the last token(s):
    ---------------------------------------
    When using chat templates (e.g., `tokenizer.apply_chat_template`), the
    sequence ends with template formatting:

        {content}<|im_end|>\n
                 ^^^^^^^^^^  ^^
                 EOS token   trailing newline (template artifact)

    The cross-entropy trajectory thus includes:
      - ce[-2]: Predicting <|im_end|> given content → MEANINGFUL SIGNAL
          * Baseline (aligned model, own text): CE ≈ 0.1 (expects ending)
          * Steered responses: CE ≈ 2-9 (surprised by abrupt ending)
      - ce[-1]: Predicting \n after <|im_end|> → NOISE (always ~20.0)
          * The model is uncertain what follows <|im_end|>
          * Could be new message, end of document, etc.

    By default, we do NOT exclude tokens (exclude_last_n=0) to keep the raw
    data intact. The \n artifact is handled at visualization time by the
    plotting code, which trims exactly 1 token when loading trajectories.

    This preserves the meaningful <|im_end|> signal while removing the noise.

    If you want to exclude at computation time, set exclude_last_n=1.

    Note: This only affects cross-entropy. Divergence/distance trajectories
    are computed on generation-time activations which don't include template
    formatting (they have length N vs cross-entropy's N+2).

    Args:
        model: Language model to evaluate with
        input_ids: [1, T] or [T] token IDs
        attention_mask: optional attention mask
        response_start_idx: index where the response starts (skip prompt tokens)
        max_value: cap for extreme values (default 20 nats ≈ ppl of 485M)
        exclude_last_n: number of tokens to exclude from the end (default 1).
            These are typically EOS/end-of-turn control tokens, not content.

    Returns:
        ce: [R] cross-entropy values for response CONTENT tokens only
            (R ≈ response_length - exclude_last_n)
            No NaN values - only content tokens are included.
            Extreme values are capped at max_value.
    """
    model.eval()

    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # [1, T, V]

    # Shift for autoregressive: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()  # [1, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()    # [1, T-1]

    # Per-token cross-entropy (in nats)
    V = shift_logits.size(-1)
    ce_per_token = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction='none'
    )  # [T-1]

    ce = ce_per_token.cpu().float().numpy()

    # Extract only response CONTENT tokens (excluding control tokens at the end)
    #
    # ce[i] = cross-entropy of predicting token[i+1] given context up to token[i]
    #
    # For response starting at response_start_idx:
    # - ce[response_start_idx - 1] = CE of predicting first response token
    # - ce[response_start_idx] = CE of predicting second response token, etc.
    #
    # We exclude the last `exclude_last_n` tokens because they are control tokens
    # (EOS, end-of-turn markers) that are NOT part of the response content.
    # Their cross-entropy is artificially high and uninformative.
    #
    # Example with exclude_last_n=1:
    #   Full response: ["Hello", ",", " how", " are", " you", "?", "<|im_end|>"]
    #   We return CE for: ["Hello", ",", " how", " are", " you", "?"]
    #   Excluded: CE for predicting "<|im_end|>"

    if response_start_idx > 0:
        start_idx = response_start_idx - 1
    else:
        start_idx = 0

    # Compute end index, ensuring we don't create negative-length arrays
    if exclude_last_n > 0:
        end_idx = len(ce) - exclude_last_n
        if end_idx <= start_idx:
            # Edge case: response too short after exclusion
            # Return at least one token if possible
            end_idx = max(start_idx + 1, len(ce))
        response_ce = ce[start_idx:end_idx]
    else:
        response_ce = ce[start_idx:]

    # Cap extreme values (but don't create inf or NaN)
    response_ce = np.clip(response_ce, 0, max_value)

    return response_ce
