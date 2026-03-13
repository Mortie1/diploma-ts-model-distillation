from __future__ import annotations

import torch

from src.metrics.base_metric import BaseMetric


class ClassificationAccuracy(BaseMetric):
    def __call__(self, logits: torch.Tensor, y: torch.Tensor, **kwargs):
        pred = logits.argmax(dim=-1)
        return (pred == y).float().mean().item()


class ForecastingMAE(BaseMetric):
    def __call__(self, forecast: torch.Tensor, target: torch.Tensor, **kwargs):
        return (forecast - target).abs().mean().item()


class ForecastingRMSE(BaseMetric):
    def __call__(self, forecast: torch.Tensor, target: torch.Tensor, **kwargs):
        return torch.sqrt(torch.mean((forecast - target) ** 2)).item()
