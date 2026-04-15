from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from src.model.adapters.lora import apply_lora


class BaseClassificationAdapter(nn.Module, ABC):
    provider_name: str = "placeholder"
    model_size_to_id: dict[str, str] = {}

    def __init__(
        self,
        model_size: str,
        n_classes: int = 2,
        in_channels: int = 1,
        require_provider_model: bool = False,
        freeze_provider: bool = True,
        fail_on_provider_fallback: bool = False,
        finetune_mode: str = "none",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_patterns: tuple[str, ...] | list[str] | str = ("*",),
    ):
        super().__init__()
        self.provider = self.provider_name
        self.model_size = model_size
        self.model_id = self._resolve_model_id(model_size)
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.require_provider_model = require_provider_model
        self.freeze_provider = freeze_provider
        self.fail_on_provider_fallback = fail_on_provider_fallback
        self.finetune_mode = str(finetune_mode).lower()

        self.provider_model = self._init_provider_model()
        self._provider_warning_printed = False

        if self.require_provider_model and self.provider_model is None:
            raise RuntimeError(
                f"Failed to initialize provider `{self.provider}` with model_id `{self.model_id}`."
            )

        self._configure_provider_finetuning(
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_patterns=lora_target_patterns,
        )

    @abstractmethod
    def _init_provider_model(self):
        raise NotImplementedError

    def _resolve_model_id(self, model_size: str) -> str:
        size_key = str(model_size).lower()
        if size_key not in self.model_size_to_id:
            supported = ", ".join(sorted(self.model_size_to_id.keys())) or "<none>"
            raise ValueError(
                f"Unsupported model_size `{model_size}` for provider `{self.provider}`. "
                f"Supported sizes: {supported}"
            )
        resolved = self.model_size_to_id[size_key]
        print(
            f"[{self.__class__.__name__}] Resolved model_size `{model_size}` -> model_id `{resolved}`."
        )
        return resolved

    def _warn_provider_fallback(self):
        if self.fail_on_provider_fallback:
            raise RuntimeError(
                f"Provider `{self.provider}` is enabled, but provider classification logits "
                "were unavailable at runtime. Aborting because fail_on_provider_fallback=true."
            )
        if not self._provider_warning_printed:
            print(
                f"[{self.__class__.__name__}] Provider `{self.provider}` unavailable or failed, "
                "no local fallback branch is available."
            )
            self._provider_warning_printed = True

    def _run_provider(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    @staticmethod
    def _normalize_target_patterns(
        patterns: tuple[str, ...] | list[str] | str,
    ) -> tuple[str, ...]:
        if isinstance(patterns, tuple):
            return patterns
        if isinstance(patterns, list):
            return tuple(str(p).strip() for p in patterns if str(p).strip())
        raw = str(patterns).strip()
        if raw.startswith("[") and raw.endswith("]"):
            raw = raw[1:-1]
        parts = [p.strip().strip("'\"") for p in raw.split(",")]
        parts = [p for p in parts if p]
        return tuple(parts) if parts else ("*",)

    def _configure_provider_finetuning(
        self,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_target_patterns: tuple[str, ...] | list[str] | str,
    ) -> None:
        if self.provider_model is None:
            return
        if not isinstance(self.provider_model, nn.Module):
            if self.finetune_mode == "none":
                return
            raise RuntimeError(
                f"Provider `{self.provider}` is not an nn.Module, finetune_mode={self.finetune_mode} is unsupported."
            )

        if self.finetune_mode == "full":
            for p in self.provider_model.parameters():
                p.requires_grad = True
            self.freeze_provider = False
            print(
                f"[{self.__class__.__name__}] finetune_mode=full: provider params are trainable."
            )
            return

        if self.finetune_mode == "lora":
            for p in self.provider_model.parameters():
                p.requires_grad = False
            target_patterns = self._normalize_target_patterns(lora_target_patterns)
            replaced = apply_lora(
                self.provider_model,
                rank=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                target_patterns=target_patterns,
            )
            if replaced <= 0:
                raise RuntimeError(
                    f"LoRA requested but no Linear layers matched target patterns {list(target_patterns)}."
                )
            self.freeze_provider = False
            print(
                f"[{self.__class__.__name__}] finetune_mode=lora: replaced {replaced} Linear layers with LoRA."
            )
            return

        if self.finetune_mode != "none":
            raise ValueError("finetune_mode must be one of: none, lora, full.")

        if self.freeze_provider:
            for p in self.provider_model.parameters():
                p.requires_grad = False

    def forward(self, inputs: torch.Tensor, **batch):
        provider_logits = self._run_provider(inputs)
        if provider_logits is None:
            self._warn_provider_fallback()
            raise RuntimeError(
                f"Provider `{self.provider}` did not return logits. "
                "This adapter is provider-only."
            )
        logits = provider_logits
        return {
            "logits": logits,
            "student_hidden": logits,
            "student_pred": logits,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            mode
            and self.freeze_provider
            and self.provider_model is not None
            and isinstance(self.provider_model, nn.Module)
        ):
            self.provider_model.eval()
        return self


class PlaceholderClassificationAdapter(BaseClassificationAdapter):
    provider_name = "placeholder"

    def _init_provider_model(self):
        return None
