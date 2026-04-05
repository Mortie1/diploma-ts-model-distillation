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
        n_classes = int(logits.shape[-1])
        f1_sum = 0.0

        for cls_idx in range(n_classes):
            cls = torch.tensor(cls_idx, device=pred.device)
            tp = ((pred == cls) & (targets == cls)).sum().item()
            fp = ((pred == cls) & (targets != cls)).sum().item()
            fn = ((pred != cls) & (targets == cls)).sum().item()

            denom = 2 * tp + fp + fn
            f1 = 0.0 if denom == 0 else (2.0 * tp) / float(denom)
            f1_sum += f1

        return f1_sum / float(n_classes)
