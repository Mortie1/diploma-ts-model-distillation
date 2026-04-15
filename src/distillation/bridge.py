from __future__ import annotations

import torch
from torch import nn

from src.distillation.teacher_audio import AudioTeacher


class DistillationBridge(nn.Module):
    """Hybrid distillation bridge for cross-modal teacher -> TS student."""

    def __init__(
        self,
        enabled: bool = False,
        input_key: str = "inputs",
        student_hidden_key: str = "student_hidden",
        student_pred_key: str = "logits",
        student_hidden_dim: int = 256,
        student_pred_dim: int = 10,
        teacher_backend: str = "mock",
        teacher_model_name: str | None = None,
        teacher_hidden_dim: int = 256,
        teacher_layer_idx: int = -1,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        self.enabled = enabled
        self.input_key = input_key
        self.student_hidden_key = student_hidden_key
        self.student_pred_key = student_pred_key

        self.teacher = AudioTeacher(
            backend=teacher_backend,
            model_name=teacher_model_name,
            hidden_dim=teacher_hidden_dim,
            layer_idx=teacher_layer_idx,
            freeze_teacher=freeze_teacher,
        )
        self.teacher_to_hidden = nn.Linear(teacher_hidden_dim, student_hidden_dim)
        self.teacher_to_pred = nn.Linear(teacher_hidden_dim, student_pred_dim)

    def forward(self, **batch):
        if not self.enabled:
            return {}

        if self.input_key not in batch:
            raise KeyError(f"Missing input key `{self.input_key}` for distillation")

        student_hidden = batch.get(self.student_hidden_key)
        student_pred = batch.get(self.student_pred_key)
        if student_hidden is None or student_pred is None:
            raise KeyError(
                "Distillation expects student outputs in batch under keys "
                f"`{self.student_hidden_key}` and `{self.student_pred_key}`"
            )

        x = batch[self.input_key]
        with torch.no_grad():
            teacher_emb = self.teacher(x)

        teacher_hidden = self.teacher_to_hidden(teacher_emb)
        teacher_pred = self.teacher_to_pred(teacher_emb)
        return {
            "teacher_hidden": teacher_hidden,
            "teacher_pred": teacher_pred,
        }
