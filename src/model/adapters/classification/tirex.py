from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class TiRexClassificationAdapter(BaseClassificationAdapter):
    provider_name = "tirex"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_head: Optional[nn.Linear] = None
        if self.provider_model is not None:
            hidden_dim = int(getattr(self.provider_model.out_norm, "weight", torch.empty(0)).numel())
            if hidden_dim > 0:
                self.provider_head = nn.Linear(hidden_dim, self.n_classes)

    def _init_provider_model(self):
        try:
            from tirex import load_model

            return load_model(self.model_id)
        except Exception:
            return None

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None or self.provider_head is None:
            return None

        series = x.mean(dim=1) if x.ndim == 3 else x
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
            return None

        return self.provider_head(features)
