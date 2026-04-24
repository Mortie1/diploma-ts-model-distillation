from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        model_name: str,
        layer_idx: int = -1,
        per_channel: bool = False,
        channel_reduce: str = "mean",
        channel_chunk_size: int = 0,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HF audio teachers. "
                "Install with `pip install transformers`"
            ) from exc

        self.model = AutoModel.from_pretrained(model_name)
        self.layer_idx = int(layer_idx)
        self.per_channel = bool(per_channel)
        self.channel_reduce = str(channel_reduce).lower()
        self.channel_chunk_size = int(channel_chunk_size)
        if self.channel_reduce not in {"mean", "max", "concat"}:
            raise ValueError("channel_reduce must be one of: mean, max, concat")

    def _pool_hidden(self, outputs) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HF speech backbones take [B, T] waveform.
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor [B, C, T], got {tuple(x.shape)}")
        if not self.per_channel:
            wav = x.mean(dim=1)
            outputs = self.model(input_values=wav, output_hidden_states=True)
            return self._pool_hidden(outputs)

        bsz, n_channels, seq_len = x.shape
        chunk = self.channel_chunk_size if self.channel_chunk_size > 0 else n_channels
        chunks = []
        for start in range(0, n_channels, chunk):
            stop = min(start + chunk, n_channels)
            c = stop - start
            wav = x[:, start:stop, :].reshape(bsz * c, seq_len)
            outputs = self.model(input_values=wav, output_hidden_states=True)
            pooled = self._pool_hidden(outputs).reshape(bsz, c, -1)  # [B, c, D]
            chunks.append(pooled)
        per_channel = torch.cat(chunks, dim=1)  # [B, C, D]
        if self.channel_reduce == "mean":
            return per_channel.mean(dim=1)
        if self.channel_reduce == "max":
            return per_channel.max(dim=1).values
        if self.channel_reduce == "concat":
            return per_channel.view(bsz, -1)


class AudioTeacher(nn.Module):
    """Factory + preprocessing wrapper for audio teachers."""

    def __init__(
        self,
        backend: str = "mock",
        model_name: Optional[str] = None,
        hidden_dim: int = 256,
        layer_idx: int = -1,
        per_channel: bool = False,
        channel_reduce: str = "mean",
        channel_chunk_size: int = 0,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        self.backend = backend

        if backend == "mock":
            self.teacher = MockAudioTeacher(hidden_dim=hidden_dim)
        elif backend == "hf":
            if model_name is None:
                raise ValueError("model_name must be set for backend='hf'")
            if hidden_dim != 256:
                logger.warning(
                    "teacher_hidden_dim=%s is ignored for backend='hf'. "
                    "HF teacher hidden size is defined by the checkpoint architecture.",
                    hidden_dim,
                )
            self.teacher = HFAudioTeacher(
                model_name=model_name,
                layer_idx=layer_idx,
                per_channel=per_channel,
                channel_reduce=channel_reduce,
                channel_chunk_size=channel_chunk_size,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher(x)
