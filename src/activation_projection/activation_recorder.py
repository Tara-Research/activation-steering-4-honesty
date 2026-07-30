from typing import Optional

import torch

from src.utils.model_utils import locate_layer


class ActivationRecorder:

    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int = -1,
        to_cpu=True
    ):
        # Store basic parameters
        self.model, self.layer_idx = model, layer_idx
        self._handle = None  # Will store the forward hook handle
        self.to_cpu = to_cpu

        self.prompt_slice: Optional[slice] = None
        self.response_slice: Optional[slice] = None

        self.response_activations = []
        self.prompt_activations = []

        self.full_activations = []

    def _hook_fn(self, module, ins, out):
        t = out[0] if isinstance(out, (tuple, list)) else out  # [B, T, H]

        # full sequence mode
        t = t.detach().cpu() if self.to_cpu else t.detach()

        # NEW: store full [B, T, H] for single-pass mapping
        self.full_activations.append(t)

        if t.shape[1] == 1:  # Single token (generation mode)
            t = t[:, -1, :]  # last position (the generated token during decode)
            t = t.detach().cpu() if self.to_cpu else t.detach()
            self.response_activations.append(t)
        else:  # Multiple tokens (e.g., full sequence encoding)
            t = t.detach().cpu() if self.to_cpu else t.detach()

            if self.prompt_slice is not None:
                prompt_t = t[:, self.prompt_slice, :]  # select only prompt positions
                self.prompt_activations.append(prompt_t)
            else:
                self.prompt_activations.append(t)

            if self.response_slice is not None:
                response_t = t[:, self.response_slice, :]  # select only response positions
                self.response_activations.append(response_t)

    # ---------- Context manager interface ----------
    def __enter__(self):
        """
        Register the forward hook when entering context.

        Returns:
            self: The steerer instance for context manager usage
        """
        layer = locate_layer(self.model, self.layer_idx, label="ActivationRecorder")
        self._handle = layer.register_forward_hook(self._hook_fn)
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

