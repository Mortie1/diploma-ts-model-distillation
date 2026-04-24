from __future__ import annotations

import torch
from torch import nn

from src.model.modules import TransformerEncoder


class StudentClassifier(nn.Module):
    """Light CNN+Transformer student for time-series classification."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hidden_dim: int = 128,
        output_dim: int = 768,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_swiglu: bool = False,
        use_rope: bool = True,
        rope_base: int = 10_000,
    ):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ff_mult=ff_mult,
            use_swiglu=use_swiglu,
            use_rope=use_rope,
            rope_base=rope_base,
        )
        self.up = nn.Linear(hidden_dim, output_dim)
        self.head = nn.Linear(output_dim, n_classes)

    def forward(self, inputs: torch.Tensor, **batch):
        # inputs: [B, C, T]
        feats = self.feature(inputs).transpose(1, 2)
        encoded = self.encoder(feats)
        pooled = encoded.mean(dim=1)
        feats = self.up(pooled)
        logits = self.head(nn.functional.gelu(feats))
        return {
            "logits": logits,
            "student_hidden": feats,
            "student_pred": logits,
        }
