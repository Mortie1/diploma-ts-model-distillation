from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.model.adapters.classification.base import BaseClassificationAdapter
from src.model.adapters.lora import apply_lora


class Chronos2ClassificationAdapter(BaseClassificationAdapter):
    provider_name = "chronos2"
    model_size_to_id = {
        "base": "amazon/chronos-2",
        "small": "amazon/chronos-2",
        "medium": "amazon/chronos-2",
        "large": "amazon/chronos-2",
    }

    def __init__(
        self,
        *args,
        finetune_mode: str = "none",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_target_patterns: tuple[str, ...]
        | list[str]
        | str = (
            "encoder.block.*.layer.*.self_attention.*",
            "encoder.block.*.layer.2.mlp.*",
        ),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.finetune_mode = str(finetune_mode).lower()
        self.provider_head: Optional[nn.Linear] = None
        if self.provider_model is not None:
            hidden_dim = int(getattr(self.provider_model, "model_dim", 0) or 0)
            if hidden_dim <= 0:
                hidden_dim = int(
                    getattr(getattr(self.provider_model, "config", None), "d_model", 0)
                    or 0
                )
            if hidden_dim > 0:
                self.provider_head = nn.Linear(hidden_dim, self.n_classes)

            if self.finetune_mode == "full":
                for p in self.provider_model.parameters():
                    p.requires_grad = True
                self.freeze_provider = False
                print(
                    "[Chronos2ClassificationAdapter] finetune_mode=full: provider params are trainable."
                )
            elif self.finetune_mode == "lora":
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
                    "[Chronos2ClassificationAdapter] finetune_mode=lora: "
                    f"replaced {replaced} Linear layers with LoRA."
                )
            elif self.finetune_mode != "none":
                raise ValueError("finetune_mode must be one of: none, lora, full.")

    def _init_provider_model(self):
        try:
            from chronos import Chronos2Pipeline

            pipeline = Chronos2Pipeline.from_pretrained(self.model_id)
            self.provider_pipeline = pipeline
            return pipeline.model
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Chronos2 provider for model_id `{self.model_id}`."
            ) from e

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

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None or self.provider_head is None:
            return None

        if x.ndim == 3:
            n_channels = int(x.shape[1])
            if n_channels != int(self.in_channels):
                raise ValueError(
                    f"Chronos2ClassificationAdapter expected inputs with "
                    f"in_channels={self.in_channels}, but got {n_channels}. "
                    "Set `model.in_channels` to match dataset channels."
                )
            batch_size, n_channels, context_len = x.shape
            # Multichannel handling: each channel is encoded as an independent
            # univariate series, and channels of the same sample share group_id.
            context = x.reshape(batch_size * n_channels, context_len)
            group_ids = (
                torch.arange(batch_size, device=x.device, dtype=torch.long)
                .unsqueeze(1)
                .expand(batch_size, n_channels)
                .reshape(-1)
            )
            needs_channel_pool = True
        elif x.ndim == 2:
            context = x
            group_ids = None
            needs_channel_pool = False
        else:
            raise ValueError(
                f"Chronos2ClassificationAdapter expects inputs with shape [B, T] or [B, C, T], got {tuple(x.shape)}."
            )

        context = context.to(dtype=torch.float32)

        def _encode() -> torch.Tensor:
            encoder_outputs, *_ = self.provider_model.encode(
                context=context,
                group_ids=group_ids,
            )
            if hasattr(encoder_outputs, "last_hidden_state"):
                return encoder_outputs.last_hidden_state
            return encoder_outputs[0]

        if self.freeze_provider:
            with torch.no_grad():
                hidden = _encode().detach()
        else:
            hidden = _encode()

        # Two-step pooling:
        # 1) temporal pooling for each (sample, channel): [B*C, Ttok, D] -> [B*C, D]
        pooled = hidden.max(dim=1).values
        # 2) channel pooling back to sample level: [B*C, D] -> [B, D]
        if needs_channel_pool:
            pooled = pooled.reshape(batch_size, n_channels, -1).max(dim=1).values
        return self.provider_head(pooled)
