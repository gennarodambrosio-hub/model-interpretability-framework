"""Activation hook manager module.
Extracts residual stream activation vectors across all layers and token positions
during model forward passes without modifying model architecture.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResidualStreamHookManager:
    """Manages registering PyTorch forward hooks to intercept hidden states from the residual stream."""

    def __init__(self, layer_modules: List[nn.Module]):
        self.layer_modules = layer_modules
        self.num_layers = len(layer_modules)
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Stores activations: Dict[layer_index -> torch.Tensor of shape (batch, seq_len, hidden_size)]
        self.captured_activations: Dict[int, torch.Tensor] = {}

    def _create_hook(self, layer_idx: int):
        def hook_fn(module: nn.Module, inputs: Tuple[Any, ...], output: Union[torch.Tensor, Tuple[Any, ...]]):
            # In transformers, layer output is typically (hidden_states, present_key_value, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store on CPU to avoid MPS/GPU memory saturation
            self.captured_activations[layer_idx] = hidden_states.detach().cpu()
        return hook_fn

    def attach_hooks(self, target_layers: Optional[List[int]] = None):
        """Attaches forward hooks to specified layers (or all layers if None)."""
        self.clear()
        layers_to_hook = target_layers if target_layers is not None else list(range(self.num_layers))

        for idx in layers_to_hook:
            if 0 <= idx < self.num_layers:
                handle = self.layer_modules[idx].register_forward_hook(self._create_hook(idx))
                self.hook_handles.append(handle)
            else:
                logger.warning(f"Layer index {idx} out of range [0, {self.num_layers - 1}]. Skipped.")

    def remove_hooks(self):
        """Removes all registered PyTorch hooks to avoid memory leaks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def clear(self):
        """Removes hooks and clears cached activations."""
        self.remove_hooks()
        self.captured_activations.clear()

    @contextmanager
    def capture(self, target_layers: Optional[List[int]] = None):
        """Context manager to safely register hooks, capture forward pass activations, and remove hooks."""
        try:
            self.attach_hooks(target_layers)
            yield self
        finally:
            self.remove_hooks()

    def get_activation(
        self,
        layer_idx: int,
        token_pos: int = -1,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """Retrieves a single activation vector of shape (hidden_size,) at a specific layer and token."""
        if layer_idx not in self.captured_activations:
            raise KeyError(f"No captured activations for layer {layer_idx}.")

        layer_acts = self.captured_activations[layer_idx]  # [batch, seq_len, d_model]
        seq_len = layer_acts.shape[1]
        
        # Support negative indexing (e.g. -1 for last token)
        actual_pos = token_pos if token_pos >= 0 else seq_len + token_pos
        if actual_pos < 0 or actual_pos >= seq_len:
            raise IndexError(f"Token position {token_pos} (resolved to {actual_pos}) out of bounds [0, {seq_len - 1}].")

        return layer_acts[batch_idx, actual_pos]  # shape: (d_model,)

    def get_layer_trajectory(
        self,
        token_pos: int = -1,
        batch_idx: int = 0,
    ) -> Dict[int, torch.Tensor]:
        """Returns the activation vector at each hooked layer for a given token position."""
        trajectory: Dict[int, torch.Tensor] = {}
        for layer_idx in sorted(self.captured_activations.keys()):
            trajectory[layer_idx] = self.get_activation(layer_idx, token_pos, batch_idx)
        return trajectory

    def compute_layer_norms(self, token_pos: int = -1) -> Dict[int, float]:
        """Computes the L2 norm of the activation across all hooked layers."""
        norms: Dict[int, float] = {}
        for layer_idx, vec in self.get_layer_trajectory(token_pos).items():
            norms[layer_idx] = float(torch.linalg.vector_norm(vec).item())
        return norms

    def compute_layer_cosine_similarities(self, token_pos: int = -1) -> Dict[str, float]:
        """Computes cosine similarity between consecutive layers for a given token."""
        trajectory = self.get_layer_trajectory(token_pos)
        sorted_layers = sorted(trajectory.keys())
        sims: Dict[str, float] = {}
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            v1, v2 = trajectory[l1], trajectory[l2]
            sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            sims[f"L{l1}->L{l2}"] = float(sim)
        return sims
