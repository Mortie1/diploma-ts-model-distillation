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


class ClassificationMacroF1(BaseMetric):
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        pred = logits.argmax(dim=-1)
        labels = torch.unique(torch.cat([targets.view(-1), pred.view(-1)]))
        if labels.numel() == 0:
            return 0.0

        f1_sum = 0.0
        for cls in labels:
            tp = ((pred == cls) & (targets == cls)).sum().item()
            fp = ((pred == cls) & (targets != cls)).sum().item()
            fn = ((pred != cls) & (targets == cls)).sum().item()

            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2.0 * tp) / float(denom)
            f1_sum += f1

        return f1_sum / float(labels.numel())


class ClassificationMicroF1(BaseMetric):
    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ):
        pred = logits.argmax(dim=-1)
        labels = torch.unique(torch.cat([targets.view(-1), pred.view(-1)]))
        if labels.numel() == 0:
            return 0.0

        tp_total = 0.0
        fp_total = 0.0
        fn_total = 0.0

        for cls in labels:
            tp_total += ((pred == cls) & (targets == cls)).sum().item()
            fp_total += ((pred == cls) & (targets != cls)).sum().item()
            fn_total += ((pred != cls) & (targets == cls)).sum().item()

        denom = 2.0 * tp_total + fp_total + fn_total
        if denom == 0.0:
            return 0.0
        return (2.0 * tp_total) / denom
