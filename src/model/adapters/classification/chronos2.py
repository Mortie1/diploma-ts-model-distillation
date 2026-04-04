from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter


class Chronos2ClassificationAdapter(BaseClassificationAdapter):
    provider_name = "chronos2"
    model_size_to_id = {
        "base": "amazon/chronos-2",
        "small": "amazon/chronos-2",
        "medium": "amazon/chronos-2",
        "large": "amazon/chronos-2",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_head: Optional[nn.Linear] = None
        if self.provider_model is not None:
            hidden_dim = int(getattr(self.provider_model, "model_dim", 0) or 0)
            if hidden_dim <= 0:
                hidden_dim = int(getattr(getattr(self.provider_model, "config", None), "d_model", 0) or 0)
            if hidden_dim > 0:
                self.provider_head = nn.Linear(hidden_dim, self.n_classes)

    def _init_provider_model(self):
        try:
            from chronos import Chronos2Pipeline
            pipeline = Chronos2Pipeline.from_pretrained(self.model_id)
            self.provider_pipeline = pipeline
            return pipeline.model
        except Exception:
            return None

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None or self.provider_head is None:
            return None

        if x.ndim == 3:
            # Chronos-2 encode path expects univariate batch [B, T].
            context = x.mean(dim=1)
        else:
            context = x

        context = context.to(dtype=torch.float32)

        def _encode() -> torch.Tensor:
            encoder_outputs, *_ = self.provider_model.encode(context=context)
            if hasattr(encoder_outputs, "last_hidden_state"):
                return encoder_outputs.last_hidden_state
            return encoder_outputs[0]

        if self.freeze_provider:
            with torch.no_grad():
                hidden = _encode().detach()
        else:
            hidden = _encode()

        pooled = hidden.mean(dim=1)
        return self.provider_head(pooled)
