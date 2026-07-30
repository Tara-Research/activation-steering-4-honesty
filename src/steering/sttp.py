from typing import Sequence, Union, Literal, Dict, Any, Optional

import numpy as np
import torch
from scipy.stats import gaussian_kde

from src.steering.threshold_config import load_threshold_config
from src.utils.model_utils import locate_layer, get_hidden_size


class SteeringToTargetProjection:
    """
    Steering to Target Projection (StTP).

    Per-layer steering along a given trait direction toward a target
    projection value derived from positive/negative training distributions.

    For each layer ℓ, we precompute:
      - a normalized steering vector v_ℓ ∈ ℝ^H,
      - projections of "negative" and "positive" training activations onto v_ℓ,
      - their means μ_neg(ℓ), μ_pos(ℓ) and stds σ_neg(ℓ), σ_pos(ℓ).

    At runtime, at a chosen layer `layer_idx`, hidden states t ∈ ℝ^{B×T×H}
    are modified so that, for the tokens we decide to steer, their projection
    ⟨t, v⟩ is snapped toward a target distribution point

        s_pos(ℓ) = μ_pos(ℓ) + coeff · σ_pos(ℓ)
        s_neg(ℓ) = μ_neg(ℓ) + coeff · σ_neg(ℓ)

    depending on `steering_direction` ('positive' or 'negative').

    Intuition:
      * steering_direction == "positive":
          raise projections that are "too low" toward the positive distribution.
      * steering_direction == "negative":
          lower projections that are "too high" toward the negative distribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: Union[torch.Tensor, Sequence[float]],
        negative_embeddings: Union[torch.Tensor, Sequence[float]],
        positive_embeddings: Union[torch.Tensor, Sequence[float]],
        *,
        bias: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        positions: Literal["all", "response", "special_tokens"] = "all",
        steering_direction: Literal["positive", "negative"] = "positive",
        coefficient: float = 0.0,
        debug: bool = False,
        register_input_capture_hook: bool = True,
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
            steering_direction:
                "positive" – push projections up toward μ_pos + coeff·σ_pos
                "negative" – push projections down toward μ_neg + coeff·σ_neg
            coefficient:
                Multiplier for the standard deviation term in the target s.
            debug:
                If True, prints basic stats about the applied deltas.
        """
        self.model, self.layer_idx = model, layer_idx
        self.positions = positions
        self.steering_direction = steering_direction
        self.coefficient = coefficient
        self.debug = debug
        self._handle = None

        self.register_input_capture_hook = register_input_capture_hook
        self.special_token_ids = [151644]  # <|im_start|>
        self._current_input_ids = None
        self._pre_handle = None
        self.response_start_idx = None  # When set, steer positions >= this idx during re-encoding

        # --- Precompute per-layer stats (μ, σ, normalized v) ---
        self.stats = self._projection_stats_all_layers(
            neg_embeddings=negative_embeddings,
            pos_embeddings=positive_embeddings,
            steering_vectors=steering_vectors,
            bias=bias,
        )

        self._v_hat_t = None
        self._threshold_t = None
        self._special_ids_t = None

        if bias is None:
            raise ValueError("[SteeringToTargetProjection] Bias is required for now to compute the threshold.")

        # Basic argument validation
        projection_param = next(model.parameters())
        v = torch.as_tensor(steering_vectors)
        if v.ndim != 2:
            raise ValueError("steering_vectors must be a 2-D tensor of shape [L, H]")

        hidden = get_hidden_size(model)
        if v.shape[1] != hidden:
            raise ValueError(
                f"Last dim of steering_vectors ({v.shape[1]}) ≠ model hidden_size ({hidden})"
            )

        if self.positions not in {"all", "response", "special_tokens"}:
            raise ValueError("positions must be 'all', 'response' or 'special_tokens'")

        if self.steering_direction not in {"positive", "negative"}:
            raise ValueError("steering_direction must be 'positive' or 'negative'")

    # ---------- Helpers ----------

    def _capture_inputs(self, module, args, kwargs):
        input_ids = None
        if kwargs is not None and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        elif len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        self._current_input_ids = input_ids

    def _kde_intersections(self, xs: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
        """
        Find x-locations where two 1D curves intersect using sign changes and
        linear interpolation between sample points.
        """
        diff = y1 - y2
        s = np.sign(diff)
        # indices where sign changes between consecutive samples
        idx = np.where(np.diff(s) != 0)[0]
        if len(idx) == 0:
            return np.array([])

        # linear interpolation for a better crossing estimate
        x0 = xs[idx]
        x1 = xs[idx + 1]
        y0 = diff[idx]
        y1v = diff[idx + 1]
        x_cross = x0 - y0 * (x1 - x0) / (y1v - y0)
        return x_cross

    def _kde_threshold(self, proj_neg: np.ndarray, proj_pos: np.ndarray, grid_pts: int = 2048):
        """
        Compute KDEs for both classes and return:
          threshold
        Threshold is chosen as the intersection closest to the mean midpoint;
        if no intersection exists, fall back to the mean midpoint.
        """
        lo = min(proj_neg.min(), proj_pos.min())
        hi = max(proj_neg.max(), proj_pos.max())
        pad = 0.05 * (hi - lo + 1e-9)
        xs = np.linspace(lo - pad, hi + pad, grid_pts)

        kde_neg = gaussian_kde(proj_neg)
        kde_pos = gaussian_kde(proj_pos)
        dens_neg = kde_neg(xs)
        dens_pos = kde_pos(xs)

        intersections = self._kde_intersections(xs, dens_neg, dens_pos)

        mid = 0.5 * (proj_neg.mean() + proj_pos.mean())
        if intersections.size:
            # pick the crossing closest to the midpoint
            threshold = intersections[np.argmin(np.abs(intersections - mid))]
        else:
            threshold = mid

        return float(threshold)

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
                threshold = self._kde_threshold(proj_neg, proj_pos)

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
        Apply distribution steering to a batch of hidden states at the hooked
        layer.

        Args:
            t:
                Hidden states at the hooked layer, shape [B, T, D].

        Behavior:
            Let v be the normalized steering vector for this layer and
            p = ⟨t, v⟩ the current projection (shape [B, T]).

            If steering_direction == 'positive':
                target s = μ_pos + coeff · σ_pos
                for positions with p < s, set their projection to s.

            If steering_direction == 'negative':
                target s = μ_neg + coeff · σ_neg
                for positions with p > s, set their projection to s.

            Concretely, for masked positions:
                t' = t - p v + s v
            i.e., we replace the component of t along v by s, leaving the
            orthogonal subspace untouched.
        """
        # Resolve layer index once
        L = len(self.stats)
        idx = self.layer_idx if self.layer_idx >= 0 else L + self.layer_idx
        layer_stats = self.stats[idx]

        # ---- Lazy, device-correct caches (critical under device_map="auto") ----
        # Cache v_hat and kde threshold on the SAME device/dtype as t
        if getattr(self, "_v_hat_t",
                   None) is None or self._v_hat_t.device != t.device or self._v_hat_t.dtype != t.dtype:
            self._v_hat_t = torch.as_tensor(layer_stats["v"], device=t.device, dtype=t.dtype)
            # cache threshold as tensor (if used)
            self._threshold_t = torch.as_tensor(layer_stats["threshold"], device=t.device, dtype=t.dtype)

        v_hat = self._v_hat_t  # (D,)

        # ---- Projection p = <t, v_hat> ----
        p = (t * v_hat).sum(dim=-1)  # (B,T)

        # ---- Target scalar s and gating mask ----
        if self.steering_direction == "positive":
            s_val = layer_stats["mu_pos"] + self.coefficient * layer_stats["std_pos"]
            s = torch.as_tensor(s_val, device=t.device, dtype=t.dtype)
            mask = p < self._threshold_t

        elif self.steering_direction == "negative":
            s_val = layer_stats["mu_neg"] + self.coefficient * layer_stats["std_neg"]
            s = torch.as_tensor(s_val, device=t.device, dtype=t.dtype)
            mask = p > self._threshold_t

        else:
            raise ValueError(f"steering_direction must be 'positive' or 'negative'. Got: {self.steering_direction}")

        # delta_p = (s - p) only where mask==True  (B,T)
        delta_p = (s - p) * mask

        # ---- positions gating without allocating (B,T,D) intermediates ----
        if self.positions == "all":
            return t + delta_p.unsqueeze(-1) * v_hat

        if self.positions == "response":
            if t.shape[1] > 1:
                if self.response_start_idx is not None:
                    delta_p[:, :self.response_start_idx] = 0.0
                    return t + delta_p.unsqueeze(-1) * v_hat
                return t  # Generation prefill: skip
            return t + delta_p.unsqueeze(-1) * v_hat

        if self.positions == "special_tokens":
            input_ids = getattr(self, "_current_input_ids", None)
            special_ids = getattr(self, "special_token_ids", None)
            if input_ids is None or not special_ids:
                return t

            if input_ids.shape[1] != t.shape[1]:
                input_ids = input_ids[:, -t.shape[1]:]

            # Cache special_ids tensor on input_ids device/dtype
            if getattr(self, "_special_ids_t",
                       None) is None or self._special_ids_t.device != input_ids.device or self._special_ids_t.dtype != input_ids.dtype:
                self._special_ids_t = torch.as_tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype)

            token_mask = torch.isin(input_ids, self._special_ids_t)  # (B,T)
            delta_p = delta_p * token_mask.to(delta_p.device)
            return t + delta_p.unsqueeze(-1) * v_hat

        raise ValueError(f"Invalid positions value: {self.positions!r}")

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
                            "[SteeringToTargetProjection] |delta| (mean ± std): "
                            f"{d.abs().mean():.4g} ± {d.std():.4g} ; "
                            f"gate_fraction={float((d.abs().sum(-1) > 0).float().mean()):.3f}"
                        )
            return new_out
        else:
            return out

    # ---------- Context manager ----------
    def __enter__(self):
        layer = locate_layer(
            self.model, self.layer_idx,
            debug=self.debug, label="SteeringToTargetProjection",
        )
        self._handle = layer.register_forward_hook(self._hook_fn)

        if self.register_input_capture_hook:
            self._pre_handle = self.model.register_forward_pre_hook(
                self._capture_inputs, with_kwargs=True
            )

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
        if self._pre_handle:
            self._pre_handle.remove()
            self._pre_handle = None


class SteeringToTargetProjectionMultipleLayers:
    """
    Convenience wrapper to apply Steering to Target Projection (StTP) on multiple layers
    at once, each with its own configuration (steering vectors, embeddings,
    coefficient, direction, positions, etc.).
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
                        "coefficient": float (optional, default 0.0),
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
        self._pre_handle = None
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
            steerer = SteeringToTargetProjection(
                model,
                steering_vectors=inst["steering_vectors"],
                negative_embeddings=inst["negative_embeddings"],
                positive_embeddings=inst["positive_embeddings"],
                bias=inst.get("bias", None),
                coefficient=inst.get("coefficient", 0.0),
                steering_direction=inst.get("steering_direction", "positive"),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                debug=debug,
                register_input_capture_hook=False,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        self._pre_handle = self.model.register_forward_pre_hook(
            self._capture_inputs, with_kwargs=True
        )

        for steerer in self._steerers:
            steerer.__enter__()
        return self

    def __exit__(self, *exc):
        self.remove()

    def _capture_inputs(self, module, args, kwargs):
        input_ids = None
        if kwargs is not None and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        elif len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        for steerer in self._steerers:
            steerer._current_input_ids = input_ids

    def remove(self):
        for steerer in self._steerers:
            steerer.remove()

        if self._pre_handle is not None:
            self._pre_handle.remove()
            self._pre_handle = None

