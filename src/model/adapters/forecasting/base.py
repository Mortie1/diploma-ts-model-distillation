from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from src.model.adapters.lora import apply_lora


class BaseForecastAdapter(nn.Module, ABC):
    """Provider-specific forecasting adapter with residual calibration head."""

    provider_name: str = "placeholder"

    def __init__(
        self,
        model_id: str,
        horizon: int,
        in_channels: int = 1,
        hidden_dim: int = 128,
        residual_scale: float = 1.0,
        require_provider_model: bool = False,
        finetune_mode: str = "none",
        gradient_checkpointing: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_patterns: tuple[str, ...] = ("*",),
    ):
        super().__init__()
        self.provider = self.provider_name
        self.model_id = model_id
        self.horizon = horizon
        self.in_channels = in_channels
        self.residual_scale = residual_scale
        self.require_provider_model = require_provider_model
        self.finetune_mode = finetune_mode.lower()
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if self.finetune_mode not in {"none", "full", "lora"}:
            raise ValueError("finetune_mode must be one of: none, full, lora")

        self.provider_model = self._init_provider_model()
        self.provider_trainable = self._extract_trainable_provider_model(self.provider_model)
        self._provider_warning_printed = False

        if self.require_provider_model and self.provider_model is None:
            raise RuntimeError(
                f"Failed to initialize provider `{self.provider}` with model_id `{self.model_id}`."
            )

        if self.finetune_mode in {"full", "lora"} and self.provider_trainable is None:
            raise RuntimeError(
                f"Provider `{self.provider}` with model `{self.model_id}` does not expose a trainable "
                "nn.Module backbone for fine-tuning."
            )

        if self.provider_trainable is not None:
            if self.finetune_mode == "none":
                for p in self.provider_trainable.parameters():
                    p.requires_grad = False
            elif self.finetune_mode == "full":
                for p in self.provider_trainable.parameters():
                    p.requires_grad = True
            else:
                for p in self.provider_trainable.parameters():
                    p.requires_grad = False
                normalized_patterns = self._normalize_lora_patterns(lora_target_patterns)
                replaced = apply_lora(
                    self.provider_trainable,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_patterns=normalized_patterns,
                )
                replaced = self._handle_lora_no_match(
                    replaced=replaced,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                if replaced == 0:
                    raise RuntimeError(
                        "LoRA requested but no Linear layers matched target patterns "
                        f"{normalized_patterns}."
                    )

        if self.gradient_checkpointing and self.provider_trainable is not None:
            self._enable_gradient_checkpointing()

        self.residual_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.residual_head = nn.Linear(hidden_dim, in_channels * horizon)

    @abstractmethod
    def _init_provider_model(self):
        raise NotImplementedError

    def _extract_trainable_provider_model(self, provider_model):
        if isinstance(provider_model, nn.Module):
            return provider_model
        return None

    def _handle_lora_no_match(
        self,
        replaced: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> int:
        return replaced

    def _enable_gradient_checkpointing(self):
        model = self.provider_trainable
        if model is None:
            return
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

    @staticmethod
    def _normalize_lora_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for pat in patterns:
            p = str(pat).strip()
            if not p:
                continue
            normalized.append(p)
            if "*" not in p and "." not in p:
                normalized.append(f"*.{p}")
        return tuple(dict.fromkeys(normalized))

    def _predict_with_trainable(
        self, context: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        return None

    @abstractmethod
    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def _warn_provider_fallback(self):
        if not self._provider_warning_printed:
            print(
                f"[{self.__class__.__name__}] Provider `{self.provider}` unavailable or failed; "
                "using trainable residual forecast only."
            )
            self._provider_warning_printed = True

    def forward(self, context: torch.Tensor, **batch):
        target = batch.get("target")
        provider_is_trainable = (
            self.provider_trainable is not None and self.finetune_mode in {"full", "lora"}
        )
        grad_ctx = nullcontext() if provider_is_trainable else torch.no_grad()

        with grad_ctx:
            base_forecast = self._predict_with_trainable(context=context, target=target)
            if base_forecast is None:
                base_forecast = self._predict_with_provider(context)

        if base_forecast is None:
            self._warn_provider_fallback()
            base_forecast = torch.zeros(
                context.size(0),
                self.in_channels,
                self.horizon,
                device=context.device,
                dtype=context.dtype,
            )

        residual_hidden = self.residual_encoder(context).squeeze(-1)
        residual = self.residual_head(residual_hidden).view(
            context.size(0), self.in_channels, self.horizon
        )
        forecast = base_forecast + self.residual_scale * residual

        hidden = context.mean(dim=-1)
        return {
            "forecast": forecast,
            "student_hidden": hidden,
            "student_pred": forecast.flatten(start_dim=1),
        }


class PlaceholderForecastAdapter(BaseForecastAdapter):
    provider_name = "placeholder"

    def _init_provider_model(self):
        return None

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        return None
