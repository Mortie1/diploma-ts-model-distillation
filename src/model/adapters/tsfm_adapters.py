from __future__ import annotations

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
        if self.finetune_mode not in {"none", "full", "lora"}:
            raise ValueError("finetune_mode must be one of: none, full, lora")

        self.provider_model = self._init_provider_model()
        self._provider_warning_printed = False
        self.provider_trainable = self.provider_model if isinstance(self.provider_model, nn.Module) else None

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
                replaced = apply_lora(
                    self.provider_trainable,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                    target_patterns=lora_target_patterns,
                )
                if replaced == 0:
                    raise RuntimeError(
                        "LoRA requested but no Linear layers matched target patterns "
                        f"{lora_target_patterns}."
                    )

        # Trainable calibrator: ensures non-empty parameter set for optimizer.
        self.residual_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.residual_head = nn.Linear(hidden_dim, in_channels * horizon)

    def _init_provider_model(self):
        if self.provider == "chronos":
            try:
                from chronos import ChronosPipeline

                return ChronosPipeline.from_pretrained(self.model_id)
            except Exception:
                return None
        if self.provider == "timesfm":
            try:
                import timesfm

                if hasattr(timesfm, "TimesFm"):
                    return timesfm.TimesFm(
                        hparams=timesfm.TimesFmHparams(
                            backend="gpu" if torch.cuda.is_available() else "cpu",
                            per_core_batch_size=32,
                            horizon_len=self.horizon,
                        ),
                        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id),
                    )
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

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        # Provider APIs are inference-style and may be non-differentiable.
        with torch.no_grad():
            if self.provider == "chronos" and self.provider_model is not None:
                series = context.mean(dim=1)  # [B, T]
                out = self.provider_model.predict(series, prediction_length=self.horizon)
                if isinstance(out, torch.Tensor):
                    # Chronos returns [B, S, H] for sample forecasts.
                    if out.ndim == 3:
                        out = out.mean(dim=1)
                    return out.unsqueeze(1).repeat(1, self.in_channels, 1)
            if self.provider == "timesfm" and self.provider_model is not None:
                # TimesFM API expects list[np.ndarray], returns point forecast.
                series = context.mean(dim=1).detach().cpu().numpy()
                inputs = [row for row in series]
                forecast, _ = self.provider_model.forecast(inputs, forecast_horizon=self.horizon)
                out = torch.tensor(forecast, dtype=context.dtype, device=context.device)
                return out.unsqueeze(1).repeat(1, self.in_channels, 1)
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

    def _warn_provider_fallback(self):
        if not self._provider_warning_printed:
            print(
                f"[TSFMForecastAdapter] Provider `{self.provider}` unavailable or failed; "
                "using trainable residual forecast only."
            )
            self._provider_warning_printed = True

    def forward(self, context: torch.Tensor, **batch):
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
