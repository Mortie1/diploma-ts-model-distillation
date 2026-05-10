from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.modules import TransformerEncoder


class StudentClassifier(nn.Module):
    """CNN+Transformer student for time-series classification.

    channel_mode:
      "independent" — each channel processed separately (channel-independent);
                      channels fused via channel_fusion (concat | mean) before head.
      "joint"       — all channels fed together as Conv1d input channels;
                      single shared representation, no channel fusion step.
    """

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
        use_cls_token: bool = True,
        channel_mode: str = "independent",
        channel_fusion: str = "concat",
        gradient_checkpointing: bool = False,
        checkpoint_every_n_layers: int = 1,
        checkpoint_use_reentrant: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.channel_mode = str(channel_mode).lower()
        self.channel_fusion = str(channel_fusion).lower()
        if self.channel_mode not in {"independent", "joint"}:
            raise ValueError("channel_mode must be one of: independent, joint.")
        if self.channel_mode == "independent" and self.channel_fusion not in {
            "concat",
            "mean",
        }:
            raise ValueError("channel_fusion must be one of: concat, mean.")
        self.use_cls_token = bool(use_cls_token)

        conv_in = 1 if self.channel_mode == "independent" else self.in_channels
        self.feature = nn.Sequential(
            nn.Conv1d(conv_in, hidden_dim, kernel_size=7, padding=3, stride=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=2),
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
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_every_n_layers=checkpoint_every_n_layers,
            checkpoint_use_reentrant=checkpoint_use_reentrant,
        )
        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, hidden_dim)) if self.use_cls_token else None
        )
        self.up = nn.Linear(hidden_dim, output_dim)
        if self.channel_mode == "joint":
            head_in = output_dim
        elif self.channel_fusion == "concat":
            head_in = output_dim * self.in_channels
        else:
            head_in = output_dim
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

        if self.channel_mode == "joint":
            # [B, C, T] → Conv1d(C, hidden_dim) → [B, hidden_dim, T'] → [B, T', hidden_dim]
            feats = self.feature(inputs).transpose(1, 2)
            if self.cls_token is not None:
                cls = self.cls_token.expand(bsz, -1, -1)
                feats = torch.cat([cls, feats], dim=1)
            encoded = self.encoder(feats)
            pooled = (
                encoded[:, 0, :] if self.cls_token is not None else encoded.mean(dim=1)
            )
            feats = self.up(pooled)
        else:
            # Channel-independent: process each channel separately, then fuse.
            x = inputs.reshape(bsz * n_channels, 1, seq_len)
            feats = self.feature(x).transpose(1, 2)
            if self.cls_token is not None:
                cls = self.cls_token.expand(feats.size(0), -1, -1)
                feats = torch.cat([cls, feats], dim=1)
            encoded = self.encoder(feats)
            pooled = (
                encoded[:, 0, :] if self.cls_token is not None else encoded.mean(dim=1)
            )
            channel_feats = self.up(pooled).reshape(bsz, n_channels, -1)
            if self.channel_fusion == "concat":
                feats = channel_feats.reshape(bsz, -1)
            else:
                feats = channel_feats.mean(dim=1)

        logits = self.head(F.gelu(feats))
        return {
            "logits": logits,
            "student_hidden": feats,
            "student_pred": logits,
        }
