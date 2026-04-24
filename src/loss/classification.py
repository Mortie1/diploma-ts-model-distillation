from __future__ import annotations

import torch
import torch.nn.functional as F
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
        feat_loss_type: str = "mse",
        feat_cosine_eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_logit = lambda_logit
        self.lambda_feat = lambda_feat
        self.temperature = temperature
        self.feat_loss_type = str(feat_loss_type).lower()
        self.feat_cosine_eps = float(feat_cosine_eps)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        if self.feat_loss_type not in {"mse", "cosine"}:
            raise ValueError(
                "feat_loss_type must be one of {'mse', 'cosine'}, "
                f"got: {self.feat_loss_type}"
            )

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
            if student_hidden.shape != teacher_hidden.shape:
                raise ValueError(
                    "student_hidden and teacher_hidden must have identical shapes for "
                    f"feature distillation. Got {tuple(student_hidden.shape)} vs "
                    f"{tuple(teacher_hidden.shape)}. "
                    "Set matching dimensions (e.g., model.output_dim and teacher hidden size)."
                )
            if self.feat_loss_type == "mse":
                feat_kd = self.mse(student_hidden, teacher_hidden)
            else:
                # Maximize cosine similarity; loss is 1 - cosine.
                s = student_hidden.reshape(student_hidden.shape[0], -1)
                t = teacher_hidden.reshape(teacher_hidden.shape[0], -1)
                s = F.normalize(s, p=2, dim=-1, eps=self.feat_cosine_eps)
                t = F.normalize(t, p=2, dim=-1, eps=self.feat_cosine_eps)
                feat_kd = (1.0 - (s * t).sum(dim=-1)).mean()

        loss = task_loss + self.lambda_logit * logit_kd + self.lambda_feat * feat_kd
        return {
            "loss": loss,
            "task_loss": task_loss,
            "logit_kd": logit_kd,
            "feat_kd": feat_kd,
        }
