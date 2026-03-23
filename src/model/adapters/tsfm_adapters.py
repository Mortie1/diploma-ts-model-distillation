from __future__ import annotations

import inspect
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from src.model.adapters.lora import apply_lora


class TSFMForecastAdapter(nn.Module):
    """
    Forecasting adapter with provider backbone + trainable calibration head.

    This allows running zero-shot style foundation forecasts (when provider libs
    are installed) and still training a residual calibrator end-to-end.
    """

    def __init__(
        self,
        provider: str,
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
        self.provider = provider.lower()
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
        self._provider_warning_printed = False
        self.provider_trainable = self._extract_trainable_provider_model(self.provider_model)

        if self.provider != "placeholder" and self.require_provider_model and self.provider_model is None:
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
            elif self.finetune_mode == "lora":
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
                if (
                    replaced == 0
                    and self.provider == "chronos"
                    and "chronos-2" in self.model_id.lower()
                ):
                    # Chronos-2 uses q/k/v/o and wi/wo naming instead of *_proj.
                    chronos2_patterns = (
                        "*.self_attention.q",
                        "*.self_attention.k",
                        "*.self_attention.v",
                        "*.self_attention.o",
                        "*.mlp.wi",
                        "*.mlp.wo",
                    )
                    replaced = apply_lora(
                        self.provider_trainable,
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        target_patterns=chronos2_patterns,
                    )
                if replaced == 0:
                    raise RuntimeError(
                        "LoRA requested but no Linear layers matched target patterns "
                        f"{normalized_patterns}."
                    )

        if self.gradient_checkpointing and self.provider_trainable is not None:
            self._enable_gradient_checkpointing()

        # Trainable calibrator: ensures non-empty parameter set for optimizer.
        self.residual_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.residual_head = nn.Linear(hidden_dim, in_channels * horizon)

    def _extract_trainable_provider_model(self, provider_model):
        if isinstance(provider_model, nn.Module):
            return provider_model
        # Chronos-2 is loaded as pipeline, but exposes trainable nn.Module at `.model`.
        if (
            self.provider == "chronos"
            and "chronos-2" in self.model_id.lower()
            and provider_model is not None
            and hasattr(provider_model, "model")
            and isinstance(provider_model.model, nn.Module)
        ):
            return provider_model.model
        return None

    def _init_provider_model(self):
        if self.provider == "chronos":
            try:
                from chronos import Chronos2Pipeline, ChronosPipeline

                pipeline_cls = ChronosPipeline
                if "chronos-2" in self.model_id.lower():
                    pipeline_cls = Chronos2Pipeline

                if torch.cuda.is_available():
                    try:
                        kwargs = {"device_map": "cuda"}
                        if pipeline_cls.__name__ == "Chronos2Pipeline":
                            kwargs["dtype"] = torch.bfloat16
                        return pipeline_cls.from_pretrained(
                            self.model_id,
                            **kwargs,
                        )
                    except Exception:
                        pass
                return pipeline_cls.from_pretrained(self.model_id)
            except Exception:
                return None
        if self.provider == "timesfm":
            try:
                import timesfm
                from huggingface_hub import hf_hub_download

                if hasattr(timesfm, "TimesFm"):
                    hparams_kwargs = {
                        "backend": "gpu" if torch.cuda.is_available() else "cpu",
                        "per_core_batch_size": 32,
                        "horizon_len": max(128, self.horizon),
                    }
                    ckpt_version = "jax"
                    try:
                        config_path = hf_hub_download(repo_id=self.model_id, filename="config.json")
                        cfg = json.loads(Path(config_path).read_text())
                        hparams_kwargs.update(
                            {
                                "context_len": int(cfg.get("context_length", 512)),
                                "horizon_len": max(int(cfg.get("horizon_length", self.horizon)), self.horizon),
                                "input_patch_len": int(cfg.get("patch_length", 32)),
                                "output_patch_len": int(cfg.get("horizon_length", max(128, self.horizon))),
                                "num_layers": int(cfg.get("num_hidden_layers", 20)),
                                "num_heads": int(cfg.get("num_attention_heads", 16)),
                                "model_dims": int(cfg.get("hidden_size", 1280)),
                                "use_positional_embedding": bool(cfg.get("use_positional_embedding", True)),
                                "quantiles": tuple(cfg.get("quantiles", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))),
                            }
                        )
                    except Exception:
                        pass
                    if "pytorch" in self.model_id.lower() or "transformers" in self.model_id.lower():
                        ckpt_version = "torch"
                    return timesfm.TimesFm(
                        hparams=timesfm.TimesFmHparams(**hparams_kwargs),
                        checkpoint=timesfm.TimesFmCheckpoint(
                            huggingface_repo_id=self.model_id,
                            version=ckpt_version,
                        ),
                    )
            except Exception:
                return None
        if self.provider in {"timesfm_hf", "timesfm_transformers"}:
            try:
                from transformers import AutoModelForTimeSeriesPrediction

                return AutoModelForTimeSeriesPrediction.from_pretrained(self.model_id)
            except Exception:
                return None
        if self.provider == "moirai":
            # Loading logic for Moirai varies across uni2ts releases.
            return None
        if self.provider == "moment":
            try:
                from momentfm import MOMENTPipeline

                model = MOMENTPipeline.from_pretrained(
                    self.model_id,
                    model_kwargs={
                        "task_name": "forecasting",
                        "forecast_horizon": self.horizon,
                    },
                )
                if hasattr(model, "init"):
                    model.init()
                return model if isinstance(model, nn.Module) else None
            except Exception:
                return None
        return None

    def _enable_gradient_checkpointing(self):
        model = self.provider_trainable
        if model is None:
            return
        # HuggingFace-style API
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        # Decoder may keep KV cache by default; disable for training memory
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
            # Most user overrides pass short names like q_proj instead of full paths.
            if "*" not in p and "." not in p:
                normalized.append(f"*.{p}")
        return tuple(dict.fromkeys(normalized))

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        provider_is_trainable = (
            self.provider_trainable is not None and self.finetune_mode in {"full", "lora"}
        )
        grad_ctx = nullcontext() if provider_is_trainable else torch.no_grad()

        # Provider APIs are inference-style and may be non-differentiable.
        with grad_ctx:
            if self.provider == "chronos" and self.provider_model is not None:
                # Chronos tokenizer internals expect CPU tensors.
                # Chronos-2 expects (B, C, T), Chronos T5 pipelines expect (B, T).
                if "chronos-2" in self.model_id.lower():
                    series = context.detach().cpu()  # [B, C, T]
                else:
                    series = context.mean(dim=1).detach().cpu()  # [B, T]
                predict_sig = inspect.signature(self.provider_model.predict)
                predict_kwargs = {
                    "prediction_length": self.horizon,
                    "limit_prediction_length": False,
                }
                # Chronos T5 pipelines accept num_samples, Chronos-2 does not.
                if "num_samples" in predict_sig.parameters:
                    predict_kwargs["num_samples"] = 1
                out = self.provider_model.predict(series, **predict_kwargs)
                if isinstance(out, torch.Tensor):
                    # Chronos returns [B, S, H] for sample forecasts.
                    if out.ndim == 3:
                        out = out.mean(dim=1)
                    out = out.to(context.device, dtype=context.dtype)
                    return out.unsqueeze(1).repeat(1, self.in_channels, 1)
                if isinstance(out, list) and len(out) > 0:
                    # Chronos-2 returns list[Tensor], typically [C, Q, H] per item.
                    rows = []
                    for item in out:
                        t = item if isinstance(item, torch.Tensor) else torch.tensor(item)
                        if t.ndim == 3:
                            # C x Q x H -> C x H using median quantile.
                            t = t[:, t.shape[1] // 2, :]
                        elif t.ndim == 2:
                            # Quantile x horizon -> use median quantile row.
                            t = t[t.shape[0] // 2]
                        elif t.ndim > 3:
                            t = t.reshape(-1, t.shape[-1]).mean(dim=0)
                        rows.append(t.to(dtype=context.dtype))
                    out_t = torch.stack(rows, dim=0).to(context.device, dtype=context.dtype)
                    if out_t.ndim == 3:
                        # [B, C, H]
                        if out_t.shape[1] != self.in_channels:
                            if out_t.shape[1] == 1:
                                out_t = out_t.repeat(1, self.in_channels, 1)
                            else:
                                out_t = out_t[:, : self.in_channels, :]
                        return out_t
                    return out_t.unsqueeze(1).repeat(1, self.in_channels, 1)
            if self.provider == "timesfm" and self.provider_model is not None:
                # TimesFM API expects list[np.ndarray], returns point forecast.
                series = context.mean(dim=1).detach().cpu().numpy()
                inputs = [row for row in series]
                forecast_kwargs = {}
                forecast_sig = inspect.signature(self.provider_model.forecast)
                if "forecast_horizon" in forecast_sig.parameters:
                    forecast_kwargs["forecast_horizon"] = self.horizon
                forecast, _ = self.provider_model.forecast(inputs, **forecast_kwargs)
                out = torch.tensor(forecast, dtype=context.dtype, device=context.device)
                if out.ndim == 2 and out.shape[-1] > self.horizon:
                    out = out[:, : self.horizon]
                return out.unsqueeze(1).repeat(1, self.in_channels, 1)
            if self.provider in {"timesfm_hf", "timesfm_transformers"} and self.provider_trainable is not None:
                # HF TimesFM expects univariate context [B, T].
                series = context.mean(dim=1)
                out = self.provider_trainable(past_values=series)
                mean_fc = out.mean_predictions
                if mean_fc.shape[-1] > self.horizon:
                    mean_fc = mean_fc[..., : self.horizon]
                elif mean_fc.shape[-1] < self.horizon:
                    pad_steps = self.horizon - mean_fc.shape[-1]
                    tail = mean_fc[..., -1:].repeat(1, pad_steps)
                    mean_fc = torch.cat([mean_fc, tail], dim=-1)
                mean_fc = mean_fc.to(context.device, dtype=context.dtype)
                return mean_fc.unsqueeze(1).repeat(1, self.in_channels, 1)
            if self.provider == "moment" and self.provider_trainable is not None:
                mask = torch.ones(
                    context.size(0), context.size(-1), dtype=torch.long, device=context.device
                )
                calls = (
                    lambda: self.provider_trainable(x_enc=context, input_mask=mask),
                    lambda: self.provider_trainable(x_enc=context),
                    lambda: self.provider_trainable(context),
                )
                for fn in calls:
                    try:
                        out = fn()
                        if isinstance(out, dict) and out.get("forecast") is not None:
                            fc = out["forecast"]
                        elif hasattr(out, "forecast") and out.forecast is not None:
                            fc = out.forecast
                        else:
                            continue
                        if fc.ndim == 2:
                            fc = fc.unsqueeze(1).repeat(1, self.in_channels, 1)
                        return fc.to(context.device, dtype=context.dtype)
                    except Exception:
                        continue

        return None

    def _predict_with_chronos2_trainable(
        self, context: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if not (
            self.provider == "chronos"
            and "chronos-2" in self.model_id.lower()
            and self.provider_trainable is not None
            and self.finetune_mode in {"full", "lora"}
        ):
            return None

        # Chronos-2 core model is univariate per sample.
        # For multivariate TS, flatten channels into the batch dimension.
        if context.ndim != 3:
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
            flat_target = target.reshape(batch_size * channels, target.shape[-1])
            kwargs["future_target"] = flat_target

        out = self.provider_trainable(**kwargs)
        quantile_preds = out.quantile_preds  # [B*C, Q, H']
        if quantile_preds is None:
            return None

        # Use median quantile as point forecast.
        median_idx = quantile_preds.shape[1] // 2
        point = quantile_preds[:, median_idx, : self.horizon]  # [B*C, H]
        if point.shape[-1] < self.horizon:
            pad_steps = self.horizon - point.shape[-1]
            point = torch.cat([point, point[:, -1:].repeat(1, pad_steps)], dim=-1)

        return point.reshape(batch_size, channels, self.horizon).to(context.device, dtype=context.dtype)

    def _warn_provider_fallback(self):
        if not self._provider_warning_printed:
            print(
                f"[TSFMForecastAdapter] Provider `{self.provider}` unavailable or failed; "
                "using trainable residual forecast only."
            )
            self._provider_warning_printed = True

    def forward(self, context: torch.Tensor, **batch):
        target = batch.get("target")
        base_forecast = self._predict_with_chronos2_trainable(context=context, target=target)
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


class TSFMClassificationAdapter(nn.Module):
    """
    Classification adapter with a trainable temporal encoder head.

    This is trainable end-to-end and can be configured with different
    provider/model ids for bookkeeping and future provider-specific wrappers.
    """

    def __init__(
        self,
        provider: str,
        model_id: str,
        n_classes: int,
        in_channels: int = 1,
        require_provider_model: bool = False,
        freeze_provider: bool = True,
    ):
        super().__init__()
        self.provider = provider.lower()
        self.model_id = model_id
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.require_provider_model = require_provider_model
        self.freeze_provider = freeze_provider
        self.provider_model = self._init_provider_model()
        self._provider_warning_printed = False
        if self.provider != "placeholder" and self.require_provider_model and self.provider_model is None:
            raise RuntimeError(
                f"Failed to initialize provider `{self.provider}` with model_id `{self.model_id}`."
            )

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def _init_provider_model(self):
        if self.provider != "moment":
            return None
        try:
            from momentfm import MOMENTPipeline

            model = MOMENTPipeline.from_pretrained(
                self.model_id,
                model_kwargs={
                    "task_name": "classification",
                    "n_channels": self.in_channels,
                    "num_class": self.n_classes,
                },
            )
            if hasattr(model, "init"):
                model.init()
            if self.freeze_provider and isinstance(model, nn.Module):
                for p in model.parameters():
                    p.requires_grad = False
            return model
        except Exception:
            return None

    def _warn_provider_fallback(self):
        if not self._provider_warning_printed:
            print(
                f"[TSFMClassificationAdapter] Provider `{self.provider}` unavailable or failed; "
                "using trainable local encoder only."
            )
            self._provider_warning_printed = True

    def _run_provider(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None:
            return None
        mask = torch.ones(x.size(0), x.size(-1), dtype=torch.long, device=x.device)
        calls = (
            lambda: self.provider_model(x_enc=x, input_mask=mask),
            lambda: self.provider_model(x_enc=x),
            lambda: self.provider_model(x),
        )
        for fn in calls:
            try:
                out = fn()
                if isinstance(out, dict) and out.get("logits") is not None:
                    return out["logits"]
                if hasattr(out, "logits") and out.logits is not None:
                    return out.logits
            except Exception:
                continue
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
