from typing import Sequence, Union, Literal, Dict, Any, Optional

import numpy as np
import torch

from src.steering.threshold_config import load_threshold_config
from src.utils.model_utils import locate_layer, get_hidden_size


class SteeringToMirrorProjection:
    """
    Steering to Mirror Projection (StMP).

    Per-layer mirror-based steering along a learned trait direction.

    For each layer ℓ, we precompute:
      - a normalized steering direction v_ℓ ∈ ℝ^H,
      - projections of "negative" and "positive" training activations onto v_ℓ,
      - their means μ_neg(ℓ), μ_pos(ℓ),
      - the midpoint m(ℓ) = 0.5 · (μ_neg(ℓ) + μ_pos(ℓ)).

    At runtime, at a chosen layer `layer_idx`, hidden states
        t ∈ ℝ^{B×T×H}
    are modified by **reflecting their projection onto v_ℓ across the midpoint m(ℓ)**.

    Let:
        p = ⟨t, v_ℓ⟩ ∈ ℝ^{B×T}

    The steering operation is always a *mirror* (reflection) in projection space:

        p' = 2·m(ℓ) − p

    Crucially, `steering_direction` only determines **which side of the midpoint
    is affected**:

      • steering_direction == "positive":
            reflect only projections on the negative side (p < m),
            i.e. mirror low-trait activations upward.

      • steering_direction == "negative":
            reflect only projections on the positive side (p > m),
            i.e. mirror high-trait activations downward.

    For tokens selected by this mask, the hidden state is updated as:

        t' = t − p·v_ℓ + p'·v_ℓ
           = t + (p' − p)·v_ℓ

    This replaces only the component of t along v_ℓ, leaving the orthogonal
    subspace unchanged.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, Sequence[float]],
        negative_embeddings: Union[torch.Tensor, Sequence[float]],
        positive_embeddings: Union[torch.Tensor, Sequence[float]],
        *,
        bias: Optional[torch.Tensor] = None,
        target_module: Optional[torch.nn.Module] = None,
        layer_idx: int = -1,
        positions: Literal["all", "response"] = "all",
        steering_direction: Literal["positive", "negative"] = "positive",
        coefficient: float = 1.0,
        debug: bool = False,
    ):
        """
        Args:
            model:
                The HF-style model whose hidden states will be steered.
            steering_vectors:
                Tensor of shape [L, H]; one steering direction per layer.
            negative_embeddings:
                Tensor of shape [L, N_neg, H]; training activations for the
                "low trait" class per layer.
            positive_embeddings:
                Tensor of shape [L, N_pos, H]; training activations for the
                "high trait" class per layer.
            layer_idx:
                Index of the transformer block to hook. Supports negative
                indexing (e.g. -1 for last layer).
            positions:
                "all"      – steer all time steps at that layer
                "response" – steer only during decoding (single-token steps),
                              skip prefill entirely
            coefficient:
                Interpolation coefficient for mirroring (default 1.0):
                  - 0.0: no change (original projection preserved)
                  - 0.5: halfway between original and mirrored position
                  - 1.0: full reflection across midpoint
                  - >1.0: overshoot past the mirrored position
            debug:
                If True, prints basic stats about the applied deltas.
        """
        self.model, self.layer_idx = model, layer_idx
        self.positions = positions
        self.steering_direction = steering_direction
        self.coefficient = coefficient
        self.coefficient_cap = 40.0
        self.debug = debug
        self._handle = None
        self.response_start_idx = None  # When set, steer positions >= this idx during re-encoding

        self.target_module = target_module

        # --- Precompute per-layer stats (μ, σ, normalized v) ---
        self.stats = self._projection_stats_all_layers(
            neg_embeddings=negative_embeddings,
            pos_embeddings=positive_embeddings,
            steering_vectors=steering_vectors,
            bias=bias,
        )

        self._v_hat_t = None
        self._threshold_t = None
        self._cap_t = None

        if bias is None:
            raise ValueError("[SteeringToMirrorProjection] Bias is required for proper threshold calculation. Please provide a bias tensor of shape [L] corresponding to the steering vectors.")

        # Basic argument validation
        v = torch.as_tensor(steering_vectors)
        if v.ndim != 2:
            raise ValueError("steering_vectors must be a 2-D tensor of shape [L, H]")

        hidden = get_hidden_size(model)
        if v.shape[1] != hidden:
            raise ValueError(
                f"Last dim of steering_vectors ({v.shape[1]}) ≠ model hidden_size ({hidden})"
            )

        if self.positions not in {"all", "response"}:
            raise ValueError("positions must be 'all' or 'response'")

        if self.steering_direction not in {"positive", "negative"}:
            raise ValueError("steering_direction must be 'positive' or 'negative'")

    # ---------- Helpers ----------

    def _resolve_layer(self):
        if self.target_module is not None:
            return self.target_module
        return locate_layer(
            self.model, self.layer_idx,
            debug=self.debug, label="SteeringToMirrorProjection",
        )

    def _mid_point_threshold(self, proj_neg: np.ndarray, proj_pos: np.ndarray, grid_pts: int = 2048):
        """
        Compute mid-point threshold between the two means of proj_neg and
        proj_pos.
        """
        mid = 0.5 * (proj_neg.mean() + proj_pos.mean())
        return mid

    def _projection_stats_all_layers(
        self,
        neg_embeddings: Union[torch.Tensor, Sequence[float]],
        pos_embeddings: Union[torch.Tensor, Sequence[float]],
        steering_vectors: Union[torch.Tensor, Sequence[float]],
        eps: float = 1e-12,
        bias: Optional[torch.Tensor] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute projection mean & std for negative and positive embeddings
        for every layer.

        Expected shapes:
            neg_embeddings:   [L, N_neg, H]
            pos_embeddings:   [L, N_pos, H]
            steering_vectors: [L, H]

        Returns:
            stats: dict[layer_index] -> {
                "proj_neg": np.ndarray [N_neg], projections of neg activations
                "proj_pos": np.ndarray [N_pos], projections of pos activations
                "mu_neg":   float,      mean proj for neg class
                "std_neg":  float,      std  proj for neg class
                "mu_pos":   float,      mean proj for pos class
                "std_pos":  float,      std  proj for pos class
                "v":        np.ndarray [H], normalized steering direction
                "threshold": float,     KDE-based threshold between classes
            }
        """
        # Convert to tensors if passed as lists/sequences
        if not isinstance(neg_embeddings, torch.Tensor):
            neg_embeddings = torch.tensor(neg_embeddings)
        if not isinstance(pos_embeddings, torch.Tensor):
            pos_embeddings = torch.tensor(pos_embeddings)
        if not isinstance(steering_vectors, torch.Tensor):
            steering_vectors = torch.tensor(steering_vectors)

        # Expect [L, N, H] for neg/pos and [L, H] for steering
        assert neg_embeddings.dim() == 3, f"neg_embeddings must be [L, N_neg, H], got {neg_embeddings.shape}"
        assert pos_embeddings.dim() == 3, f"pos_embeddings must be [L, N_pos, H], got {pos_embeddings.shape}"
        assert steering_vectors.dim() == 2, f"steering_vectors must be [L, H], got {steering_vectors.shape}"

        L, _, H = neg_embeddings.shape
        assert steering_vectors.shape == (L, H), (
            f"steering_vectors must have shape [L, H], got {steering_vectors.shape}"
        )

        stats: Dict[int, Dict[str, Any]] = {}

        threshold_cfg = load_threshold_config()

        for layer in range(L):
            # [N_neg, H], [N_pos, H]
            neg_layer = neg_embeddings[layer].detach().cpu().numpy()
            pos_layer = pos_embeddings[layer].detach().cpu().numpy()

            # [H]
            v = steering_vectors[layer].detach().cpu().numpy()

            # Capture ‖w_stored‖ before normalizing to unit vector.
            # For logistic regression with projection normalization, the bias
            # has been pre-rescaled by the same factor as the weights, so
            # threshold = -b_rescaled / ‖w_stored‖ = -b_original / ‖w_original‖.
            v_norm = np.linalg.norm(v) + eps

            # normalize steering direction
            v = v / v_norm

            # Projections: [N_neg], [N_pos]
            proj_neg = neg_layer @ v
            proj_pos = pos_layer @ v

            # Mean & std of projections
            mu_neg = float(proj_neg.mean())
            std_neg = float(proj_neg.std(ddof=1)) if proj_neg.shape[0] > 1 else 0.0
            mu_pos = float(proj_pos.mean())
            std_pos = float(proj_pos.std(ddof=1)) if proj_pos.shape[0] > 1 else 0.0

            if threshold_cfg["mode"] == "neg_mean_std":
                threshold = float(mu_neg + threshold_cfg["alpha"] * std_neg)
            elif bias is not None:
                threshold = float(-bias[layer].item() / v_norm)
            else:
                threshold = self._mid_point_threshold(proj_neg, proj_pos)

            stats[layer] = {
                "proj_neg": proj_neg,
                "proj_pos": proj_pos,
                "mu_neg": mu_neg,
                "std_neg": std_neg,
                "mu_pos": mu_pos,
                "std_pos": std_pos,
                "v": v,  # normalized steering direction
                "threshold": threshold,
            }

        return stats

    def _apply_steering(self, t: torch.Tensor) -> torch.Tensor:
        """
        Mirror-steering around the midpoint of the two means.

        - Always uses reflection p' = 2*m - p.
        - If steering_direction == "positive": apply only when p < m.
        - If steering_direction == "negative": apply only when p > m.

        Args:
            t: Hidden states at the hooked layer, shape [B, T, D].
        """
        if self._v_hat_t is None or self._v_hat_t.device != t.device or self._v_hat_t.dtype != t.dtype:
            idx = self.layer_idx if self.layer_idx >= 0 else len(self.stats) + self.layer_idx
            self._v_hat_t = torch.as_tensor(
                self.stats[idx]["v"],
                device=t.device,
                dtype=t.dtype,
            )
            self._threshold_t = torch.as_tensor(
                self.stats[idx]["threshold"],
                device=t.device,
                dtype=t.dtype,
            )
            cap_val = self.stats[idx]["mu_pos"] + self.coefficient_cap * self.stats[idx]["std_pos"]
            self._cap_t = torch.as_tensor(cap_val, device=t.device, dtype=t.dtype)

        v_hat = self._v_hat_t  # (D,) on correct device/dtype
        threshold = self._threshold_t  # scalar tensor on correct device/dtype

        p = (t * v_hat).sum(dim=-1)  # (B,T)

        if self.steering_direction == "positive":
            mask = p < threshold
        else:
            mask = p > threshold

        # Compute mirror target: p_mirror = 2*threshold - p
        # Then interpolate: p' = p + coef * (p_mirror - p)
        #                      = p + coef * (2*threshold - 2p)
        # When coef=0: p' = p (no change)
        # When coef=1: p' = 2*threshold - p (full mirror)
        # When coef>1: overshoot past mirror

        mirror_target = 2 * threshold - p  # (B,T)
        p_prime = p + self.coefficient * (mirror_target - p)  # (B,T)
        p_prime = torch.clamp(p_prime, max=self._cap_t)
        delta_p = (p_prime - p)  # (B,T)
        delta_p = delta_p * mask  # (B,T)

        # --- ADD THIS SANITY CHECK ---
        if torch.isnan(delta_p).any() or torch.isinf(delta_p).any():
            delta_p = torch.nan_to_num(delta_p, nan=0.0, posinf=0.0, neginf=0.0)

        if self.positions == "all":
            return t + delta_p.unsqueeze(-1) * v_hat

        elif self.positions == "response":
            # Steer only during decoding (single-token generation steps).
            # During prefill (t.shape[1] > 1), return unchanged so that
            # the prompt's KV cache remains unsteered.
            if t.shape[1] > 1:
                if self.response_start_idx is not None:
                    delta_p[:, :self.response_start_idx] = 0.0
                    return t + delta_p.unsqueeze(-1) * v_hat
                return t
            return t + delta_p.unsqueeze(-1) * v_hat

    def _hook_fn(self, module, ins, out):
        """
        Forward hook to apply steering to the output of the hooked layer.

        Supports either:
          * a single tensor output (standard HF models), or
          * a tuple/list with the hidden states as the first element.
        """
        if torch.is_tensor(out):
            return self._apply_steering(out)
        elif isinstance(out, (tuple, list)):
            head = out[0]
            if not torch.is_tensor(head):
                return out
            head2 = self._apply_steering(head)
            new_out = (head2, *out[1:])

            if self.debug:
                with torch.no_grad():
                    d = head2 - head
                    if d.numel() > 0:
                        print(
                            "[SteeringToMirrorProjection] |delta| (mean ± std): "
                            f"{d.abs().mean():.4g} ± {d.std():.4g} ; "
                            f"gate_fraction={float((d.abs().sum(-1) > 0).float().mean()):.3f}"
                        )
            return new_out
        else:
            return out

    # ---------- Context manager ----------

    def __enter__(self):
        """
        Register the forward hook on the specified transformer layer so that
        steering is applied automatically during forward passes.
        """
        layer = self._resolve_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        """
        Remove the forward hook when leaving the context manager.
        """
        self.remove()

    def remove(self):
        """
        Explicitly remove the registered forward hook (if any).
        """
        if self._handle:
            self._handle.remove()
            self._handle = None


class SteeringToMirrorProjectionMultipleLayers:
    """
    Convenience wrapper to apply Steering to Mirror Projection (StMP) on multiple layers
    at once, each with its own configuration (steering vectors, embeddings,
    direction, positions, etc.).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        """
        Args:
            model:
                The HF-style model to be steered.
            instructions:
                A sequence of dictionaries, one per steered layer, each with:
                    {
                        "steering_vectors": [L, H] tensor,
                        "negative_embeddings": [L, N_neg, H] tensor,
                        "positive_embeddings": [L, N_pos, H] tensor,
                        "layer_idx": int,
                        "steering_direction": "positive" or "negative" (optional),
                        "positions": "all" | "response" (optional),
                    }
            debug:
                If True, each underlying steerer will print debug info.
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []   # Store all hook handles for cleanup
        self._steerers = []  # Individual steerer instances

        if not instructions:
            return

        # Pre-compute stats ONCE using the first instruction's embeddings/vectors.
        # ASSUMPTION: All instructions share the same embeddings, just target different layers.
        # This is validated below.
        first_inst = instructions[0]

        # Validate that all instructions share the same embeddings (stats depend only on these)
        for i, inst in enumerate(instructions[1:], start=1):
            if inst["steering_vectors"] is not first_inst["steering_vectors"]:
                raise ValueError(
                    f"Instruction {i} has different steering_vectors than instruction 0. "
                    "All instructions must share the same embeddings for stats sharing to work."
                )
            if inst["negative_embeddings"] is not first_inst["negative_embeddings"]:
                raise ValueError(
                    f"Instruction {i} has different negative_embeddings than instruction 0. "
                    "All instructions must share the same embeddings for stats sharing to work."
                )
            if inst["positive_embeddings"] is not first_inst["positive_embeddings"]:
                raise ValueError(
                    f"Instruction {i} has different positive_embeddings than instruction 0. "
                    "All instructions must share the same embeddings for stats sharing to work."
                )

        # Create individual steerers for each instruction, each computes its own stats
        for inst in self.instructions:
            steerer = SteeringToMirrorProjection(
                model,
                steering_vectors=inst["steering_vectors"],
                negative_embeddings=inst["negative_embeddings"],
                positive_embeddings=inst["positive_embeddings"],
                bias=inst.get("bias", None),
                steering_direction=inst.get("steering_direction", "positive"),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                coefficient=inst.get("coefficient", 1.0),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        """
        Register forward hooks for all configured steerers when entering
        the context manager.
        """
        for steerer in self._steerers:
            layer = steerer._resolve_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        """
        Clean up all hooks when exiting the context.
        """
        self.remove()

    def remove(self):
        """
        Remove all registered forward hooks and clean up resources.
        """
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()
