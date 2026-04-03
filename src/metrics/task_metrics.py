from __future__ import annotations

import torch

from src.metrics.base_metric import BaseMetric


class ClassificationAccuracy(BaseMetric):
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        pred = logits.argmax(dim=-1)
        return (pred == targets).float().mean().item()


class ForecastingMAE(BaseMetric):
    def __call__(
        self,
        forecast: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        return (forecast - targets).abs().mean().item()


class ForecastingRMSE(BaseMetric):
    def __call__(
        self,
        forecast: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        return torch.sqrt(torch.mean((forecast - targets) ** 2)).item()
