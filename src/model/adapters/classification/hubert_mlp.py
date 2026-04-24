from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class HuBERTMLPClassificationAdapter(BaseClassificationAdapter):
    """
    HuBERT feature extractor + MLP classifier for time-series classification.

    Multichannel inputs are encoded per-channel by treating channels as extra
    batch elements, then fused back at sample level.
    """

    provider_name = "hubert_mlp"
    model_size_to_id = {
        "base": "facebook/hubert-base-ls960",
        "large": "facebook/hubert-large-ll60k",
    }

    def __init__(
        self,
        *args,
        teacher_layer_idx: int = 1,
        channel_fusion: str = "concat",
        temporal_pool: str = "mean",
        mlp_hidden_dim: int = 512,
        mlp_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_layer_idx = int(teacher_layer_idx)
        self.channel_fusion = str(channel_fusion).lower()
        if self.channel_fusion not in {"mean", "concat"}:
            raise ValueError("channel_fusion must be one of: mean, concat.")
        self.temporal_pool = str(temporal_pool).lower()
        if self.temporal_pool not in {"mean", "max"}:
            raise ValueError("temporal_pool must be one of: mean, max.")

        self.provider_head: Optional[nn.Module] = None
        if self.provider_model is not None:
            hidden_dim = int(
                getattr(getattr(self.provider_model, "config", None), "hidden_size", 0)
            )
            if hidden_dim <= 0:
                raise RuntimeError("Failed to infer HuBERT hidden_size from config.")
            head_in = (
                hidden_dim * int(self.in_channels)
                if self.channel_fusion == "concat"
                else hidden_dim
            )
            self.provider_head = nn.Sequential(
                nn.Linear(head_in, int(mlp_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(mlp_dropout)),
                nn.Linear(int(mlp_hidden_dim), self.n_classes),
            )

    def _init_provider_model(self):
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(self.model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize HuBERT provider for model_id `{self.model_id}`."
            ) from e

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None or self.provider_head is None:
            return None

        if x.ndim == 3:
            bsz, n_channels, seq_len = x.shape
            if n_channels != int(self.in_channels):
                raise ValueError(
                    f"HuBERTMLPClassificationAdapter expected in_channels={self.in_channels}, "
                    f"got {n_channels}."
                )
            wav = x.reshape(bsz * n_channels, seq_len)
            needs_channel_pool = True
        elif x.ndim == 2:
            wav = x
            bsz = int(x.shape[0])
            n_channels = 1
            needs_channel_pool = False
        else:
            raise ValueError(
                f"HuBERTMLPClassificationAdapter expects input [B, C, T] or [B, T], got {tuple(x.shape)}."
            )

        wav = wav.to(dtype=torch.float32)

        def _encode():
            out = self.provider_model(input_values=wav, output_hidden_states=True)
            if self.teacher_layer_idx == -1:
                hidden = out.last_hidden_state
            else:
                hs = out.hidden_states
                if hs is None:
                    raise RuntimeError(
                        "HuBERT did not return hidden_states while teacher_layer_idx != -1."
                    )
                idx = self.teacher_layer_idx
                if idx < 0:
                    idx = len(hs) + idx
                if idx < 0 or idx >= len(hs):
                    raise IndexError(
                        f"teacher_layer_idx={self.teacher_layer_idx} out of range for {len(hs)} hidden states."
                    )
                hidden = hs[idx]
            return hidden

        if self.freeze_provider:
            with torch.no_grad():
                hidden = _encode().detach()
        else:
            hidden = _encode()

        # Pool over time.
        if self.temporal_pool == "max":
            pooled = hidden.max(dim=1).values
        else:
            pooled = hidden.mean(dim=1)  # [B*C, D] or [B, D]

        # Fuse channels back to sample-level.
        if needs_channel_pool:
            pooled = pooled.reshape(bsz, n_channels, -1)
            if self.channel_fusion == "mean":
                pooled = pooled.mean(dim=1)
            else:
                pooled = pooled.reshape(bsz, -1)
        elif self.channel_fusion == "concat" and int(self.in_channels) != 1:
            raise ValueError(
                "Received [B, T] input while channel_fusion=concat and in_channels>1."
            )

        return self.provider_head(pooled)
