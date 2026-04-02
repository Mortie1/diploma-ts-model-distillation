from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class BaseClassificationAdapter(nn.Module, ABC):
    provider_name: str = "placeholder"

    def __init__(
        self,
        model_id: str,
        n_classes: int,
        in_channels: int = 1,
        require_provider_model: bool = False,
        freeze_provider: bool = True,
        fail_on_provider_fallback: bool = False,
    ):
        super().__init__()
        self.provider = self.provider_name
        self.model_id = model_id
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.require_provider_model = require_provider_model
        self.freeze_provider = freeze_provider
        self.fail_on_provider_fallback = fail_on_provider_fallback

        self.provider_model = self._init_provider_model()
        self._provider_warning_printed = False

        if self.require_provider_model and self.provider_model is None:
            raise RuntimeError(
                f"Failed to initialize provider `{self.provider}` with model_id `{self.model_id}`."
            )

        if self.freeze_provider and self.provider_model is not None and isinstance(self.provider_model, nn.Module):
            for p in self.provider_model.parameters():
                p.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    @abstractmethod
    def _init_provider_model(self):
        raise NotImplementedError

    def _warn_provider_fallback(self):
        if self.fail_on_provider_fallback:
            raise RuntimeError(
                f"Provider `{self.provider}` is enabled, but provider classification logits "
                "were unavailable at runtime. Aborting because fail_on_provider_fallback=true."
            )
        if not self._provider_warning_printed:
            print(
                f"[{self.__class__.__name__}] Provider `{self.provider}` unavailable or failed; "
                "using trainable local encoder only."
            )
            self._provider_warning_printed = True

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def forward(self, x: torch.Tensor, **batch):
        pooled = self.encoder(x).squeeze(-1)
        logits = self.head(pooled)
        provider_logits = self._run_provider(x)
        if provider_logits is None:
            self._warn_provider_fallback()
        else:
            logits = logits + provider_logits.to(logits.device, dtype=logits.dtype)
        return {
            "logits": logits,
            "student_hidden": pooled,
            "student_pred": logits,
        }


class PlaceholderClassificationAdapter(BaseClassificationAdapter):
    provider_name = "placeholder"

    def _init_provider_model(self):
        return None
