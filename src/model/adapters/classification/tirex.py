from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class TiRexClassificationAdapter(BaseClassificationAdapter):
    provider_name = "tirex"
    model_size_to_id = {
        "base": "NX-AI/TiRex",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_fusion = str(kwargs.get("channel_fusion", "concat")).lower()
        if self.channel_fusion not in {"mean", "concat"}:
            raise ValueError("channel_fusion must be one of: mean, concat.")
        self.provider_head: Optional[nn.Linear] = None
        if self.provider_model is not None:
            hidden_dim = int(
                getattr(self.provider_model.out_norm, "weight", torch.empty(0)).numel()
            )
            if hidden_dim > 0:
                head_in = (
                    hidden_dim * int(self.in_channels)
                    if self.channel_fusion == "concat"
                    else hidden_dim
                )
                self.provider_head = nn.Linear(head_in, self.n_classes)

    def _init_provider_model(self):
        try:
            from tirex import load_model

            return load_model(self.model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TiRex provider for model_id `{self.model_id}`."
            ) from e

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None or self.provider_head is None:
            return None

        if x.ndim == 3:
            bsz, n_channels, seq_len = x.shape
            if n_channels != int(self.in_channels):
                raise ValueError(
                    f"TiRexClassificationAdapter expected in_channels={self.in_channels}, "
                    f"got {n_channels}."
                )
            series = x.reshape(bsz * n_channels, seq_len)
            needs_channel_pool = True
        elif x.ndim == 2:
            bsz, seq_len = x.shape
            n_channels = 1
            series = x
            needs_channel_pool = False
        else:
            raise ValueError(f"Expected input rank 2/3, got shape={tuple(x.shape)}")

        series = series.to(dtype=torch.float32)

        if self.freeze_provider:
            with torch.no_grad():
                embeds = self.provider_model._embed_context(series).detach()
        else:
            embeds = self.provider_model._embed_context(series)

        if embeds.ndim == 4:
            # [B, tokens, layers, hidden] -> last layer, mean over tokens.
            features = embeds[:, :, -1, :].mean(dim=1)
        elif embeds.ndim == 3:
            # [B, tokens, hidden]
            features = embeds.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unexpected TiRex embedding shape: {tuple(embeds.shape)}. "
                "Expected rank 3 or 4 tensor."
            )

        if needs_channel_pool:
            features = features.reshape(bsz, n_channels, -1)
            if self.channel_fusion == "mean":
                features = features.mean(dim=1)
            else:
                features = features.reshape(bsz, -1)
        elif self.channel_fusion == "concat" and int(self.in_channels) != 1:
            raise ValueError(
                "Received [B, T] input while channel_fusion=concat and in_channels>1."
            )

        return self.provider_head(features)
