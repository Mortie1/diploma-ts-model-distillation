from __future__ import annotations

import logging

import torch
from torch import nn

from src.distillation.teacher_audio import AudioTeacher

logger = logging.getLogger(__name__)


class DistillationBridge(nn.Module):
    """Thin distillation bridge: extract teacher hidden representations."""

    def __init__(
        self,
        enabled: bool = False,
        input_key: str = "inputs",
        teacher_backend: str = "mock",
        teacher_model_name: str | None = None,
        teacher_hidden_dim: int = 128,
        teacher_layer_idx: int = -1,
        teacher_per_channel: bool = False,
        teacher_channel_reduce: str = "mean",
        teacher_channel_chunk_size: int = 0,
        freeze_teacher: bool = True,
        teacher_projection: str = "none",
        teacher_pca_dim: int = 768,
        teacher_pca_fit_batches: int = 30,
        teacher_pca_center: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.input_key = input_key

        self.teacher = None
        if self.enabled:
            self.teacher = AudioTeacher(
                backend=teacher_backend,
                model_name=teacher_model_name,
                hidden_dim=teacher_hidden_dim,
                layer_idx=teacher_layer_idx,
                per_channel=teacher_per_channel,
                channel_reduce=teacher_channel_reduce,
                channel_chunk_size=teacher_channel_chunk_size,
                freeze_teacher=freeze_teacher,
            )
        self.teacher_projection = str(teacher_projection).lower()
        if self.teacher_projection not in {"none", "pca"}:
            raise ValueError("teacher_projection must be one of: none, pca")
        self.teacher_pca_dim = int(teacher_pca_dim)
        self.teacher_pca_fit_batches = int(teacher_pca_fit_batches)
        self.teacher_pca_center = bool(teacher_pca_center)
        if self.teacher_projection == "pca" and self.teacher_pca_dim <= 0:
            raise ValueError("teacher_pca_dim must be > 0 when teacher_projection=pca")

        # PCA state for teacher projection.
        self.register_buffer("teacher_pca_mean", torch.empty(0))
        self.register_buffer("teacher_pca_components", torch.empty(0, 0))
        self.register_buffer("teacher_pca_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer(
            "teacher_pca_seen_batches", torch.tensor(0, dtype=torch.long)
        )
        self._teacher_pca_fit_cache: list[torch.Tensor] = []

    def _fit_teacher_pca(self, teacher_emb: torch.Tensor) -> bool:
        """
        Accumulate teacher embeddings and fit PCA once enough batches are seen.
        Returns True when PCA is ready to project embeddings.
        """
        if self.teacher_projection != "pca":
            return False
        if bool(self.teacher_pca_ready.item()):
            return True
        if teacher_emb.ndim != 2:
            raise ValueError(
                f"PCA projection expects teacher_emb of shape [B, D], got {tuple(teacher_emb.shape)}"
            )

        self._teacher_pca_fit_cache.append(teacher_emb.detach().float().cpu())
        self.teacher_pca_seen_batches.add_(1)

        fit_after = max(1, self.teacher_pca_fit_batches)
        if int(self.teacher_pca_seen_batches.item()) < fit_after:
            return False

        x = torch.cat(self._teacher_pca_fit_cache, dim=0)  # [N, D]
        self._teacher_pca_fit_cache.clear()
        if self.teacher_pca_center:
            mean = x.mean(dim=0, keepdim=True)
            x_centered = x - mean
        else:
            mean = torch.zeros((1, x.shape[1]), dtype=x.dtype)
            x_centered = x

        n_samples, in_dim = x_centered.shape
        out_dim = min(self.teacher_pca_dim, int(in_dim), int(n_samples))
        if out_dim <= 0:
            raise RuntimeError(
                f"Cannot fit PCA with shape={tuple(x_centered.shape)} and "
                f"teacher_pca_dim={self.teacher_pca_dim}."
            )

        _, _, v = torch.pca_lowrank(x_centered, q=out_dim, center=False)
        components = v[:, :out_dim].contiguous()  # [D, K]

        self.teacher_pca_mean = mean.squeeze(0)
        self.teacher_pca_components = components
        self.teacher_pca_ready.fill_(True)
        logger.info(
            "Fitted teacher PCA: samples=%d in_dim=%d out_dim=%d batches=%d",
            n_samples,
            in_dim,
            out_dim,
            int(self.teacher_pca_seen_batches.item()),
        )
        return True

    def _apply_teacher_projection(
        self, teacher_emb: torch.Tensor
    ) -> torch.Tensor | None:
        if self.teacher_projection == "none":
            return teacher_emb

        if self.teacher_projection == "pca":
            ready = self._fit_teacher_pca(teacher_emb)
            if not ready:
                return None
            mean = self.teacher_pca_mean.to(
                device=teacher_emb.device, dtype=teacher_emb.dtype
            )
            components = self.teacher_pca_components.to(
                device=teacher_emb.device, dtype=teacher_emb.dtype
            )
            return (teacher_emb - mean.unsqueeze(0)) @ components

        raise RuntimeError(f"Unexpected teacher_projection: {self.teacher_projection}")

    def forward(self, **batch):
        if not self.enabled:
            return {}
        if self.teacher is None:
            raise RuntimeError(
                "Distillation is enabled, but teacher is not initialized."
            )

        if self.input_key not in batch:
            raise KeyError(f"Missing input key `{self.input_key}` for distillation")

        x = batch[self.input_key]
        with torch.no_grad():
            teacher_emb = self.teacher(x)
            teacher_emb = self._apply_teacher_projection(teacher_emb)

        if teacher_emb is None:
            # PCA warmup phase (not fitted yet): skip feature distillation for this batch.
            return {}

        return {"teacher_hidden": teacher_emb}
