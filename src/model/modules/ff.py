from typing import Literal

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_dim_multiplier: int,
        activation: Literal["relu", "gelu"],
        dropout_p: float,
    ):
        super().__init__()

        match activation:
            case "relu":
                self.activation = nn.ReLU()
            case "gelu":
                self.activation = nn.GELU()
            case _:
                raise ValueError("activation must be one of: 'relu', 'gelu'")
        inner_dim = hidden_dim * hidden_dim_multiplier
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            self.activation,
            nn.Dropout(dropout_p),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor):
        return self.ff(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        hidden_dim_multiplier: int,
        dropout_p: float,
    ):
        super().__init__()

        inner_dim = hidden_dim * hidden_dim_multiplier
        self.up = nn.Linear(hidden_dim, inner_dim)

        self.gate = nn.Sequential(nn.Linear(hidden_dim, inner_dim), nn.SiLU())
        self.dropout1 = nn.Dropout(dropout_p)
        self.proj = nn.Linear(inner_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        x_up = self.up(x)
        x_gate = self.gate(x)

        return self.dropout2(self.proj(self.dropout1(x_up * x_gate)))
