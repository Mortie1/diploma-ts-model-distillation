from __future__ import annotations

import torch
from torch import nn


class ForecastingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        forecast: torch.Tensor,
        targets: torch.Tensor,
        **batch,
    ):
        loss = self.mse(forecast, targets)
        return {"loss": loss, "task_loss": loss}


class DistillForecastingLoss(nn.Module):
    def __init__(
        self,
        lambda_logit: float = 0.5,
        lambda_feat: float = 0.5,
    ):
        super().__init__()
        self.lambda_logit = lambda_logit
        self.lambda_feat = lambda_feat
        self.mse = nn.MSELoss()

    def forward(
        self,
        forecast: torch.Tensor,
        student_hidden: torch.Tensor,
        student_pred: torch.Tensor,
        targets: torch.Tensor,
        teacher_hidden: torch.Tensor | None = None,
        teacher_pred: torch.Tensor | None = None,
        **batch,
    ):
        task_loss = self.mse(forecast, targets)

        logit_kd = torch.tensor(0.0, device=forecast.device)
        feat_kd = torch.tensor(0.0, device=forecast.device)

        if teacher_pred is not None:
            logit_kd = self.mse(student_pred, teacher_pred)

        if teacher_hidden is not None:
            feat_kd = self.mse(student_hidden, teacher_hidden)

        loss = task_loss + self.lambda_logit * logit_kd + self.lambda_feat * feat_kd
        return {
            "loss": loss,
            "task_loss": task_loss,
            "logit_kd": logit_kd,
            "feat_kd": feat_kd,
        }
