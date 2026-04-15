from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class Chronos2ClassificationAdapter(BaseClassificationAdapter):
    provider_name = "chronos2"
    model_size_to_id = {"base": "amazon/chronos-2"}

    def __init__(
        self,
        *args,
        channel_embed_dim: int = 256,
        channel_fusion: str = "concat",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.channel_embed_dim = int(channel_embed_dim)
        self.channel_fusion = str(channel_fusion).lower()
        self.provider_head: Optional[nn.Linear] = None
        self.provider_proj: Optional[nn.Module] = None
        if self.provider_model is not None:
            hidden_dim = int(getattr(self.provider_model, "model_dim", 0) or 0)
            if hidden_dim <= 0:
                hidden_dim = int(
                    getattr(getattr(self.provider_model, "config", None), "d_model", 0)
                    or 0
                )
            if hidden_dim > 0:
                if self.channel_embed_dim <= 0:
                    raise ValueError("channel_embed_dim must be > 0.")
                self.provider_proj = nn.Linear(hidden_dim, self.channel_embed_dim)
                if self.channel_fusion == "concat":
                    head_in = self.channel_embed_dim * int(self.in_channels)
                elif self.channel_fusion in {"mean", "max"}:
                    head_in = self.channel_embed_dim
                else:
                    raise ValueError(
                        "channel_fusion must be one of: concat, mean, max."
                    )
                self.provider_head = nn.Linear(head_in, self.n_classes)

    def _init_provider_model(self):
        try:
            from chronos import Chronos2Pipeline

            pipeline = Chronos2Pipeline.from_pretrained(self.model_id)
            self.provider_pipeline = pipeline
            return pipeline.model
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Chronos2 provider for model_id `{self.model_id}`."
            ) from e

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            self.provider_model is None
            or self.provider_head is None
            or self.provider_proj is None
        ):
            return None

        if x.ndim == 3:
            n_channels = int(x.shape[1])
            if n_channels != int(self.in_channels):
                raise ValueError(
                    f"Chronos2ClassificationAdapter expected inputs with "
                    f"in_channels={self.in_channels}, but got {n_channels}. "
                    "Set `model.in_channels` to match dataset channels."
                )
            batch_size, n_channels, context_len = x.shape
            # Multichannel handling: each channel is encoded as an independent
            # univariate series, and channels of the same sample share group_id.
            context = x.reshape(batch_size * n_channels, context_len)
            # group_ids = (
            #     torch.arange(batch_size, device=x.device, dtype=torch.long)
            #     .unsqueeze(1)
            #     .expand(batch_size, n_channels)
            #     .reshape(-1)
            # )
            group_ids = None
            needs_channel_pool = True
        elif x.ndim == 2:
            context = x
            group_ids = None
            needs_channel_pool = False
        else:
            raise ValueError(
                f"Chronos2ClassificationAdapter expects inputs with shape [B, T] or [B, C, T], got {tuple(x.shape)}."
            )

        context = context.to(dtype=torch.float32)

        def _encode() -> torch.Tensor:
            encoder_outputs, *_ = self.provider_model.encode(
                context=context,
                group_ids=group_ids,
            )
            if hasattr(encoder_outputs, "last_hidden_state"):
                return encoder_outputs.last_hidden_state
            return encoder_outputs[0]

        if self.freeze_provider:
            with torch.no_grad():
                hidden = _encode().detach()
        else:
            hidden = _encode()

        # For concat fusion, concatenate channel embeddings into one large vector.
        # We do not concatenate along time: use last token per channel.
        # For non-concat fusion, also use last token representation.
        if self.channel_fusion == "concat":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden[:, -1, :]

        # Project channel embeddings (e.g., 768 -> 512/256).
        pooled = self.provider_proj(pooled)

        # Channel fusion back to sample level.
        if needs_channel_pool:
            pooled = pooled.reshape(batch_size, n_channels, -1)
            if self.channel_fusion == "concat":
                pooled = pooled.reshape(batch_size, -1)
            elif self.channel_fusion == "mean":
                pooled = pooled.mean(dim=1)
            elif self.channel_fusion == "max":
                pooled = pooled.max(dim=1).values
            else:
                raise ValueError("channel_fusion must be one of: concat, mean, max.")
        elif self.channel_fusion == "concat" and int(self.in_channels) != 1:
            raise ValueError(
                "Received univariate input [B, T] while channel_fusion=concat and "
                f"in_channels={self.in_channels}. Expected [B, C, T] input."
            )
        return self.provider_head(pooled)
