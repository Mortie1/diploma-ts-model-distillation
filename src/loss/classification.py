from __future__ import annotations

import torch
from torch import nn


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **batch,
    ):
        loss = self.ce(logits, targets)
        return {"loss": loss, "task_loss": loss}


class DistillClassificationLoss(nn.Module):
    def __init__(
        self,
        lambda_logit: float = 0.5,
        lambda_feat: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.lambda_logit = lambda_logit
        self.lambda_feat = lambda_feat
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        logits: torch.Tensor,
        student_hidden: torch.Tensor,
        targets: torch.Tensor,
        teacher_hidden: torch.Tensor | None = None,
        teacher_pred: torch.Tensor | None = None,
        **batch,
    ):
        task_loss = self.ce(logits, targets)

        logit_kd = torch.tensor(0.0, device=logits.device)
        feat_kd = torch.tensor(0.0, device=logits.device)

        if teacher_pred is not None:
            t = self.temperature
            student_log_probs = torch.log_softmax(logits / t, dim=-1)
            teacher_probs = torch.softmax(teacher_pred / t, dim=-1)
            logit_kd = self.kl(student_log_probs, teacher_probs) * (t * t)

        if teacher_hidden is not None:
            feat_kd = self.mse(student_hidden, teacher_hidden)

        loss = task_loss + self.lambda_logit * logit_kd + self.lambda_feat * feat_kd
        return {
            "loss": loss,
            "task_loss": task_loss,
            "logit_kd": logit_kd,
            "feat_kd": feat_kd,
        }
