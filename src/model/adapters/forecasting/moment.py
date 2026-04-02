from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.forecasting.base import BaseForecastAdapter


class MomentForecastAdapter(BaseForecastAdapter):
    provider_name = "moment"

    def _init_provider_model(self):
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
            return model
        except Exception:
            return None

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_trainable is None:
            return None
        mask = torch.ones(context.size(0), context.size(-1), dtype=torch.long, device=context.device)
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
