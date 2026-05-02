from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class MantisClassificationAdapter(BaseClassificationAdapter):
    provider_name = "mantis"
    model_size_to_id = {
        "v1": "paris-noah/Mantis-8M",
        "v2": "paris-noah/MantisV2",
        "base": "paris-noah/MantisV2",
    }

    def __init__(
        self,
        *args,
        channel_embed_dim: int = 256,
        channel_fusion: str = "concat",
        return_transf_layer: int = 2,
        output_token: str = "combined",
        **kwargs,
    ):
        self.return_transf_layer = int(return_transf_layer)
        self.output_token = str(output_token)
        super().__init__(*args, **kwargs)

        self.channel_embed_dim = int(channel_embed_dim)
        self.channel_fusion = str(channel_fusion).lower()
        self.provider_proj: Optional[nn.Module] = None
        self.provider_head: Optional[nn.Linear] = None
        self._is_v2 = "mantisv2" in self.model_id.lower()

        if self.provider_model is not None:
            hidden_dim = int(getattr(self.provider_model, "hidden_dim", 0) or 0)
            if hidden_dim > 0:
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
            if "mantisv2" in self.model_id.lower():
                from mantis.architecture.version2 import MantisV2 as MantisModel
            else:
                from mantis.architecture.version1 import MantisV1 as MantisModel

            model = MantisModel(
                return_transf_layer=self.return_transf_layer,
                output_token=self.output_token,
                device="cpu",
            ).from_pretrained(self.model_id)
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Mantis provider for model_id `{self.model_id}`."
            ) from e

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            self.provider_model is None
            or self.provider_proj is None
            or self.provider_head is None
        ):
            return None

        if x.ndim == 3:
            batch_size, n_channels, context_len = x.shape
            if n_channels != int(self.in_channels):
                raise ValueError(
                    f"MantisV2ClassificationAdapter expected inputs with "
                    f"in_channels={self.in_channels}, but got {n_channels}. "
                    "Set `model.in_channels` to match dataset channels."
                )
            context = x.reshape(batch_size * n_channels, 1, context_len)
            needs_channel_pool = True
        elif x.ndim == 2:
            batch_size = x.size(0)
            n_channels = 1
            context = x.unsqueeze(1)
            needs_channel_pool = False
        else:
            raise ValueError(
                f"MantisClassificationAdapter expects [B, T] or [B, C, T], got {tuple(x.shape)}."
            )

        provider_device = next(self.provider_model.parameters()).device
        context = context.to(device=provider_device, dtype=torch.float32)

        num_patches = int(getattr(self.provider_model, "num_patches", 32) or 32)
        seq_len = int(context.size(-1))
        remainder = seq_len % num_patches
        if remainder != 0:
            target_len = seq_len + (num_patches - remainder)
            context = torch.nn.functional.interpolate(
                context,
                size=target_len,
                mode="linear",
                align_corners=False,
            )

        if self.freeze_provider:
            with torch.no_grad():
                hidden = self.provider_model(context).detach()
        else:
            hidden = self.provider_model(context)

        if hidden.ndim != 2:
            raise RuntimeError(
                f"Unexpected Mantis output shape: {tuple(hidden.shape)}. Expected [B, hidden]."
            )

        pooled = self.provider_proj(hidden)
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


# Backward-compatible alias for existing configs/CLI overrides.
class MantisV2ClassificationAdapter(MantisClassificationAdapter):
    pass
