from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class MockAudioTeacher(nn.Module):
    """Small fallback teacher for offline/smoke runs."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, 1, T]
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, C, T], got {tuple(x.shape)}")
        x = x.mean(dim=1, keepdim=True)
        z = self.encoder(x)
        return z.mean(dim=-1)


class HFAudioTeacher(nn.Module):
    """Audio foundation-model wrapper via HuggingFace Transformers."""

    def __init__(self, model_name: str, hidden_dim: int, layer_idx: int = -1):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HF audio teachers. "
                "Install with `pip install transformers`"
            ) from exc

        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.layer_idx = int(layer_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF speech backbones take [B, T] waveform.
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, C, T], got {tuple(x.shape)}")
        wav = x.mean(dim=1)
        outputs = self.model(input_values=wav, output_hidden_states=True)
        hidden = None
        if self.layer_idx == -1:
            hidden = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError(
                    "HF teacher did not return hidden_states while layer_idx != -1."
                )
            n_layers = len(hidden_states)
            idx = self.layer_idx
            if idx < 0:
                idx = n_layers + idx
            if idx < 0 or idx >= n_layers:
                raise IndexError(
                    f"teacher_layer_idx={self.layer_idx} is out of range for "
                    f"{n_layers} hidden states."
                )
            hidden = hidden_states[idx]
        return hidden.mean(dim=1)


class AudioTeacher(nn.Module):
    """Factory + preprocessing wrapper for audio teachers."""

    def __init__(
        self,
        backend: str = "mock",
        model_name: Optional[str] = None,
        hidden_dim: int = 256,
        layer_idx: int = -1,
        freeze_teacher: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.backend = backend

        if backend == "mock":
            self.teacher = MockAudioTeacher(hidden_dim=hidden_dim)
        elif backend == "hf":
            if model_name is None:
                raise ValueError("model_name must be set for backend='hf'")
            self.teacher = HFAudioTeacher(
                model_name=model_name,
                hidden_dim=hidden_dim,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Per-sample z-score normalization.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        return self.teacher(x)
