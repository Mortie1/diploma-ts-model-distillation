from __future__ import annotations

import torch
from torch import nn


class TSFMForecastAdapter(nn.Module):
    """Adapter shell for external forecasting TSFMs (TimesFM/Chronos/Moirai)."""

    def __init__(self, provider: str, model_id: str, horizon: int, in_channels: int = 1):
        super().__init__()
        self.provider = provider
        self.model_id = model_id
        self.horizon = horizon
        self.in_channels = in_channels

    def forward(self, context: torch.Tensor, **batch):
        # Placeholder for real model integration. Produces naive baseline
        # to keep pipeline executable before provider-specific hookup.
        last = context[:, :, -1:]
        forecast = last.repeat(1, 1, self.horizon)
        hidden = context.mean(dim=-1)
        return {
            "forecast": forecast,
            "student_hidden": hidden,
            "student_pred": forecast.flatten(start_dim=1),
        }


class TSFMClassificationAdapter(nn.Module):
    """Adapter shell for external classification-capable TSFMs (e.g. MOMENT)."""

    def __init__(self, provider: str, model_id: str, n_classes: int, in_channels: int = 1):
        super().__init__()
        self.provider = provider
        self.model_id = model_id
        self.in_channels = in_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, n_classes)

    def forward(self, x: torch.Tensor, **batch):
        pooled = self.pool(x).squeeze(-1)
        logits = self.head(pooled)
        return {
            "logits": logits,
            "student_hidden": pooled,
            "student_pred": logits,
        }
