"""MLP for grokking experiments."""

import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "quadratic": None,  # handled manually
}


class MLP(nn.Module):
    """Simple MLP with configurable depth and activation.

    Args:
        in_size: Input dimension (2p for one-hot encoded pairs).
        hidden_size: Width of each hidden layer.
        out_size: Output dimension (p classes).
        depth: Number of hidden layers (default 1 matches original paper).
        activation: "relu", "gelu", "tanh", or "quadratic" (original paper).
        dropout: Dropout rate applied after each activation.
        bias: Whether to use bias terms.
    """

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        out_size: int,
        depth: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.activation_name = activation

        layers = []
        prev = in_size
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden_size, bias=bias))
            if activation == "quadratic":
                layers.append(_Quadratic())
            else:
                layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_size
        layers.append(nn.Linear(prev, out_size, bias=bias))

        self.net = nn.Sequential(*layers)

        # Keep fc1/fc2 references for analysis code that reads weight matrices
        self.fc1 = self.net[0]
        self.fc2 = self.net[-1]

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name == "quadratic":
                    scale = 0.25**0.5 / np.power(2 * m.weight.shape[0], 1 / 3)
                    nn.init.normal_(m.weight, mean=0.0, std=scale)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


class _Quadratic(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x**2
