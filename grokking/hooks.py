"""Forward/backward hooks for capturing intermediate activations."""

import torch
import torch.nn as nn


class Hook:
    """Captures the output (and input) of a module during forward or backward pass."""

    def __init__(self, module: nn.Module, backward: bool = False):
        fn = module.register_full_backward_hook if backward else module.register_forward_hook
        self.hook = fn(self._hook_fn)
        self.input = None
        self.output = None

    def _hook_fn(self, module: nn.Module, input: tuple, output: torch.Tensor):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()
