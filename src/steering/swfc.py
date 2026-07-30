# swfc.py
# Steering with Fixed Coefficient (SwFC).
# Provides classes for steering transformer model activations by adding a
# fixed-coefficient-scaled vector to specific layers during forward passes.

from typing import Sequence, Union

import torch

from src.utils.model_utils import locate_layer, get_hidden_size


class SteeringWithFixedCoef:
    """
    Steering with Fixed Coefficient (SwFC).

    Add (coef * steering_vector) to a chosen transformer block's output.

    This class modifies model behavior by injecting a steering vector into the activations
    at a specified layer during the forward pass. Supports different position modes for
    selective steering (prompt-only, response-only, or all positions).

    Handles blocks that return tuples and fails loudly if it can't locate a layer list.

    Args:
        model: The transformer model to steer
        steering_vector: 1D tensor or sequence representing the steering direction
        coeff: Scaling coefficient for the steering vector (default: 1.0)
        layer_idx: Index of the layer to hook (supports negative indexing, default: -1)
        positions: Where to apply steering - 'all', 'response', or 'special_tokens' (default: 'all')
        debug: Enable debug output showing steering magnitude (default: False)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
        register_input_capture_hook: bool = True,
    ):
        # Store basic parameters
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None  # Will store the forward hook handle
        self._pre_handle = None
        self.response_start_idx = None  # When set, steer positions >= this idx during re-encoding

        self.register_input_capture_hook = register_input_capture_hook

        # --- Convert steering vector to tensor on model's device/dtype ---
        # Get a sample parameter to determine device and dtype
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)

        # Validate vector is 1-dimensional
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")

        hidden = get_hidden_size(model)
        if self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )

        # Validate positions parameter
        if self.positions not in {"all", "response", "special_tokens"}:
            raise ValueError("positions must be 'all', 'response' or 'special_tokens'")

    # ---------- Helper methods ----------
    def _capture_inputs(self, module, args, kwargs):
        input_ids = None
        if kwargs is not None and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
        elif len(args) > 0 and torch.is_tensor(args[0]):
            input_ids = args[0]

        self._current_input_ids = input_ids

    def _hook_fn(self, module, ins, out):
        """
        Forward hook function that adds the steering vector to layer outputs.

        This hook is called during the forward pass and modifies the hidden states
        by adding a scaled steering vector. Behavior depends on self.positions:
        - 'all': Add steering to all token positions
        - 'response': Steer only during decoding (single-token steps), skip prefill entirely

        Handles both tensor outputs and tuple/list outputs from transformer blocks.

        Args:
            module: The layer module being hooked
            ins: Input tensors (unused)
            out: Output from the layer (tensor or tuple of tensors)

        Returns:
            Modified output with steering vector added
        """
        # Scale the steering vector by coefficient
        steer = self.coeff * self.vector  # (hidden,)

        def _add(t):
            """Apply steering based on position setting"""
            if self.positions == "all":
                # Add steering to all positions
                return t + steer.to(t.device)
            elif self.positions == "response":
                # Steer only during decoding (single-token generation steps).
                # During prefill (t.shape[1] > 1), return unchanged so that
                # the prompt's KV cache remains unsteered.
                if t.shape[1] > 1:
                    if self.response_start_idx is not None:
                        steered = t.clone()
                        steered[:, self.response_start_idx:, :] += steer.to(t.device)
                        return steered
                    return t  # Generation prefill: skip
                return t + steer.to(t.device)  # Generation decode: single token

            elif self.positions == "special_tokens":
                input_ids = getattr(self, "_current_input_ids", None)
                special_ids = getattr(self, "special_token_ids", None)

                # If we cannot identify tokens, do nothing (fail-safe)
                if input_ids is None or special_ids is None or len(special_ids) == 0:
                    return t

                # Align captured input_ids to the time dimension of `t`
                # (handles generation steps where t.shape[1] == 1)
                if input_ids.shape[1] != t.shape[1]:
                    input_ids = input_ids[:, -t.shape[1]:]

                # Compute token mask: True where token id is a "special token"
                special_ids_t = torch.as_tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype)
                token_mask = torch.isin(input_ids, special_ids_t)  # (B, T)

                # Restrict the existing projection-based mask to special-token positions
                mask_special = token_mask.to(t.device)  # (B, T)

                # Apply steering only at those positions
                steering = steer * mask_special.unsqueeze(-1)

                return t + steering

            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # Handle different output types: tensor or tuple/list
        if torch.is_tensor(out):
            new_out = _add(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # Unusual case – don't modify
                return out
            # Apply steering to first element (hidden states), keep rest unchanged
            head = _add(out[0])
            new_out = (head, *out[1:])
        else:
            return out  # Unknown type – leave unchanged

        # Debug output: show magnitude of changes
        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out

    # ---------- Context manager interface ----------
    def __enter__(self):
        """
        Register the forward hook when entering context.

        Returns:
            self: The steerer instance for context manager usage
        """
        layer = locate_layer(
            self.model, self.layer_idx,
            debug=self.debug, label="ActivationSteerer",
        )
        self._handle = layer.register_forward_hook(self._hook_fn)
        if self.register_input_capture_hook:
            self._pre_handle = self.model.register_forward_pre_hook(
                self._capture_inputs, with_kwargs=True
            )
        return self

    def __exit__(self, *exc):
        """
        Clean up hook when exiting context.

        Args:
            *exc: Exception information (if any)
        """
        self.remove()

    def remove(self):
        """Remove the registered forward hook and clean up resources."""
        if self._handle:
            self._handle.remove()
            self._handle = None
        if self._pre_handle:
            self._pre_handle.remove()
            self._pre_handle = None


class SteeringWithFixedCoefMultiple:
    """
    Add multiple (coef * steering_vector) to chosen transformer block outputs.

    This class allows steering multiple layers simultaneously with different steering
    vectors and coefficients. Useful for complex multi-layer interventions.

    Args:
        model: The transformer model to steer
        instructions: List of dicts, each containing:
            - 'steering_vector': The vector to add
            - 'coef': Scaling coefficient (optional, default: 0.0)
            - 'layer_idx': Layer index to hook (optional, default: -1)
            - 'positions': Position mode (optional, default: 'all')
        debug: Enable debug output (default: False)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []  # Store all hook handles for cleanup
        self._steerers = []  # Store individual steerer instances

        # Create individual steerers for each instruction
        for inst in self.instructions:
            steerer = SteeringWithFixedCoef(
                model,
                inst["steering_vector"],
                coeff=inst["coeff"],
                layer_idx=inst["layer_idx"],
                positions=inst["positions"],
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        """
        Register all forward hooks when entering context.

        Returns:
            self: The steerer instance for context manager usage
        """
        for steerer in self._steerers:
            layer = locate_layer(
                steerer.model, steerer.layer_idx,
                debug=steerer.debug, label="ActivationSteerer",
            )
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        """
        Clean up all hooks when exiting context.

        Args:
            *exc: Exception information (if any)
        """
        self.remove()

    def remove(self):
        """Remove all registered forward hooks and clean up resources."""
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()
