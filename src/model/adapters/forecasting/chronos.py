from __future__ import annotations

import inspect
import math
from typing import Optional

import torch
from torch import nn

from src.model.adapters.forecasting.base import BaseForecastAdapter
from src.model.adapters.lora import apply_lora


class ChronosForecastAdapter(BaseForecastAdapter):
    provider_name = "chronos"

    def _init_provider_model(self):
        try:
            from chronos import Chronos2Pipeline, ChronosPipeline

            pipeline_cls = Chronos2Pipeline if "chronos-2" in self.model_id.lower() else ChronosPipeline

            if torch.cuda.is_available():
                try:
                    kwargs = {"device_map": "cuda"}
                    if pipeline_cls.__name__ == "Chronos2Pipeline":
                        kwargs["dtype"] = torch.bfloat16
                    return pipeline_cls.from_pretrained(self.model_id, **kwargs)
                except Exception:
                    pass
            return pipeline_cls.from_pretrained(self.model_id)
        except Exception:
            return None

    def _extract_trainable_provider_model(self, provider_model):
        if isinstance(provider_model, nn.Module):
            return provider_model
        if (
            "chronos-2" in self.model_id.lower()
            and provider_model is not None
            and hasattr(provider_model, "model")
            and isinstance(provider_model.model, nn.Module)
        ):
            return provider_model.model
        return None

    def _handle_lora_no_match(
        self,
        replaced: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
    ) -> int:
        if replaced > 0 or self.provider_trainable is None or "chronos-2" not in self.model_id.lower():
            return replaced
        # Chronos-2 uses q/k/v/o and wi/wo naming instead of *_proj.
        patterns = (
            "*.self_attention.q",
            "*.self_attention.k",
            "*.self_attention.v",
            "*.self_attention.o",
            "*.mlp.wi",
            "*.mlp.wo",
        )
        return apply_lora(
            self.provider_trainable,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_patterns=patterns,
        )

    def _predict_with_trainable(
        self, context: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if not (
            "chronos-2" in self.model_id.lower()
            and self.provider_trainable is not None
            and self.finetune_mode in {"full", "lora"}
            and context.ndim == 3
        ):
            return None

        batch_size, channels, context_len = context.shape
        flat_context = context.reshape(batch_size * channels, context_len)
        flat_context_mask = torch.ones_like(flat_context, dtype=flat_context.dtype, device=flat_context.device)
        group_ids = torch.arange(batch_size * channels, device=context.device, dtype=torch.long)

        output_patch_size = int(self.provider_trainable.chronos_config.output_patch_size)
        num_output_patches = max(1, math.ceil(self.horizon / output_patch_size))

        kwargs = {
            "context": flat_context,
            "context_mask": flat_context_mask,
            "group_ids": group_ids,
            "num_output_patches": num_output_patches,
        }
        if target is not None and target.ndim == 3:
            kwargs["future_target"] = target.reshape(batch_size * channels, target.shape[-1])

        out = self.provider_trainable(**kwargs)
        quantile_preds = out.quantile_preds
        if quantile_preds is None:
            return None

        median_idx = quantile_preds.shape[1] // 2
        point = quantile_preds[:, median_idx, : self.horizon]
        if point.shape[-1] < self.horizon:
            pad_steps = self.horizon - point.shape[-1]
            point = torch.cat([point, point[:, -1:].repeat(1, pad_steps)], dim=-1)

        return point.reshape(batch_size, channels, self.horizon).to(context.device, dtype=context.dtype)

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None:
            return None

        # Chronos pipeline.predict builds internal DataLoader and expects CPU tensors.
        if "chronos-2" in self.model_id.lower():
            series = context.detach().cpu()  # [B, C, T]
        else:
            series = context.mean(dim=1).detach().cpu()  # [B, T]

        predict_sig = inspect.signature(self.provider_model.predict)
        predict_kwargs = {
            "prediction_length": self.horizon,
            "limit_prediction_length": False,
        }
        if "num_samples" in predict_sig.parameters:
            predict_kwargs["num_samples"] = 1

        out = self.provider_model.predict(series, **predict_kwargs)
        if isinstance(out, torch.Tensor):
            if out.ndim == 3:
                out = out.mean(dim=1)
            out = out.to(context.device, dtype=context.dtype)
            return out.unsqueeze(1).repeat(1, self.in_channels, 1)

        if isinstance(out, list) and len(out) > 0:
            rows = []
            for item in out:
                t = item if isinstance(item, torch.Tensor) else torch.tensor(item)
                if t.ndim == 3:
                    t = t[:, t.shape[1] // 2, :]
                elif t.ndim == 2:
                    t = t[t.shape[0] // 2]
                elif t.ndim > 3:
                    t = t.reshape(-1, t.shape[-1]).mean(dim=0)
                rows.append(t.to(dtype=context.dtype))
            out_t = torch.stack(rows, dim=0).to(context.device, dtype=context.dtype)
            if out_t.ndim == 3:
                if out_t.shape[1] != self.in_channels:
                    if out_t.shape[1] == 1:
                        out_t = out_t.repeat(1, self.in_channels, 1)
                    else:
                        out_t = out_t[:, : self.in_channels, :]
                return out_t
            return out_t.unsqueeze(1).repeat(1, self.in_channels, 1)

        return None
