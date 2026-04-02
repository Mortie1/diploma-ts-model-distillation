from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Optional

import torch

from src.model.adapters.forecasting.base import BaseForecastAdapter


class TimesFMForecastAdapter(BaseForecastAdapter):
    provider_name = "timesfm"

    def _init_provider_model(self):
        try:
            import timesfm
            from huggingface_hub import hf_hub_download

            if not hasattr(timesfm, "TimesFm"):
                return None

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

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_model is None:
            return None
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
