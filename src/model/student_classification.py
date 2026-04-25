from __future__ import annotations

import torch
from torch import nn

from src.model.modules import TransformerEncoder


class StudentClassifier(nn.Module):
    """Channel-independent CNN+Transformer student for time-series classification."""

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
        channel_fusion: str = "concat",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.channel_fusion = str(channel_fusion).lower()
        if self.channel_fusion not in {"concat", "mean"}:
            raise ValueError("channel_fusion must be one of: concat, mean.")

        # Channel-independent frontend: one shared encoder applied to each channel.
        self.feature = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=7, padding=3),
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
        head_in = (
            output_dim * self.in_channels
            if self.channel_fusion == "concat"
            else output_dim
        )
        self.head = nn.Linear(head_in, n_classes)

    def forward(self, inputs: torch.Tensor, **batch):
        # inputs: [B, C, T] (preferred) or [B, T] (single channel).
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError(
                f"StudentClassifier expects inputs [B, C, T] or [B, T], got {tuple(inputs.shape)}."
            )
        bsz, n_channels, seq_len = inputs.shape
        if n_channels != self.in_channels:
            raise ValueError(
                f"StudentClassifier expected in_channels={self.in_channels}, got {n_channels}."
            )

        x = inputs.reshape(bsz * n_channels, 1, seq_len)
        feats = self.feature(x).transpose(1, 2)
        encoded = self.encoder(feats)
        pooled = encoded.mean(dim=1)
        channel_feats = self.up(pooled).reshape(bsz, n_channels, -1)
        if self.channel_fusion == "concat":
            feats = channel_feats.reshape(bsz, -1)
        else:
            feats = channel_feats.mean(dim=1)
        logits = self.head(nn.functional.gelu(feats))
        return {
            "logits": logits,
            "student_hidden": feats,
            "student_pred": logits,
        }
