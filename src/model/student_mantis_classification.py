from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.model.modules import TransformerEncoder


class _ScalarEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, P, 1]
        return self.net(x)


class MantisStudentClassifier(nn.Module):
    """Student with Mantis-style token generator before Transformer."""

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
        channel_fusion: str = "concat",
        num_patches: int = 32,
        tokgen_kernel_size: int = 41,
        scalar_hidden_dim: int = 32,
        tokgen_eps: float = 1e-5,
        gradient_checkpointing: bool = False,
        checkpoint_every_n_layers: int = 1,
        checkpoint_use_reentrant: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_patches = int(num_patches)
        self.tokgen_eps = float(tokgen_eps)
        self.use_cls_token = bool(use_cls_token)
        self.channel_fusion = str(channel_fusion).lower()
        if self.channel_fusion not in {"concat", "mean"}:
            raise ValueError("channel_fusion must be one of: concat, mean.")
        if self.num_patches <= 0:
            raise ValueError("num_patches must be > 0.")

        k = int(tokgen_kernel_size)
        if k <= 0:
            raise ValueError("tokgen_kernel_size must be > 0.")
        if k % 2 == 0:
            k += 1

        # Mantis-like token generator branches: x and diff(x).
        self.conv_x = nn.Conv1d(1, self.hidden_dim, kernel_size=k, padding=k // 2)
        self.conv_diff = nn.Conv1d(1, self.hidden_dim, kernel_size=k, padding=k // 2)
        self.ln_x = nn.LayerNorm(self.hidden_dim)
        self.ln_diff = nn.LayerNorm(self.hidden_dim)

        self.mean_encoder = _ScalarEncoder(out_dim=int(scalar_hidden_dim))
        self.std_encoder = _ScalarEncoder(out_dim=int(scalar_hidden_dim))
        self.token_projector = nn.Linear(
            2 * self.hidden_dim + 2 * int(scalar_hidden_dim),
            self.hidden_dim,
        )

        self.encoder = TransformerEncoder(
            hidden_dim=self.hidden_dim,
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
            nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            if self.use_cls_token
            else None
        )
        self.up = nn.Linear(self.hidden_dim, self.output_dim)
        head_in = (
            self.output_dim * self.in_channels
            if self.channel_fusion == "concat"
            else self.output_dim
        )
        self.head = nn.Linear(head_in, self.n_classes)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.tokgen_eps)
        return (x - mu) / std

    def _to_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, T]
        n, _, t = x.shape
        rem = t % self.num_patches
        if rem != 0:
            target_t = t + (self.num_patches - rem)
            x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)
            t = target_t

        x_norm = self._normalize(x)
        diff = torch.diff(x, dim=-1)
        diff = F.pad(diff, (0, 1))
        diff_norm = self._normalize(diff)

        x_feat = self.conv_x(x_norm)  # [N, H, T]
        d_feat = self.conv_diff(diff_norm)  # [N, H, T]

        x_feat = self.ln_x(x_feat.transpose(1, 2)).transpose(1, 2)
        d_feat = self.ln_diff(d_feat.transpose(1, 2)).transpose(1, 2)

        patch_len = t // self.num_patches
        x_patch = x_feat.reshape(n, self.hidden_dim, self.num_patches, patch_len).mean(
            dim=-1
        )
        d_patch = d_feat.reshape(n, self.hidden_dim, self.num_patches, patch_len).mean(
            dim=-1
        )
        x_patch = x_patch.transpose(1, 2)  # [N, P, H]
        d_patch = d_patch.transpose(1, 2)  # [N, P, H]

        x_patched = x.reshape(n, self.num_patches, patch_len)
        patch_mean = x_patched.mean(dim=-1, keepdim=True)  # [N, P, 1]
        patch_std = x_patched.std(dim=-1, keepdim=True).clamp_min(self.tokgen_eps)
        mean_emb = self.mean_encoder(patch_mean)  # [N, P, S]
        std_emb = self.std_encoder(patch_std)  # [N, P, S]

        tokens = torch.cat([d_patch, x_patch, mean_emb, std_emb], dim=-1)
        return self.token_projector(tokens)  # [N, P, H]

    def forward(self, inputs: torch.Tensor, **batch):
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 3:
            raise ValueError(
                f"MantisStudentClassifier expects [B, C, T] or [B, T], got {tuple(inputs.shape)}."
            )
        bsz, n_channels, seq_len = inputs.shape
        if n_channels != self.in_channels:
            raise ValueError(
                f"MantisStudentClassifier expected in_channels={self.in_channels}, got {n_channels}."
            )

        x = inputs.reshape(bsz * n_channels, 1, seq_len).to(dtype=torch.float32)
        tokens = self._to_patch_tokens(x)  # [B*C, P, H]
        if self.cls_token is not None:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        encoded = self.encoder(tokens)
        pooled = encoded[:, 0, :] if self.cls_token is not None else encoded.mean(dim=1)
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
