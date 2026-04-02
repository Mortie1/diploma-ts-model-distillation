from __future__ import annotations

from typing import Optional

import torch

from src.model.adapters.forecasting.base import BaseForecastAdapter


class TimesFMHFForecastAdapter(BaseForecastAdapter):
    provider_name = "timesfm_hf"

    def _init_provider_model(self):
        try:
            from transformers import AutoModelForTimeSeriesPrediction

            return AutoModelForTimeSeriesPrediction.from_pretrained(self.model_id)
        except Exception:
            return None

    def _predict_with_provider(self, context: torch.Tensor) -> Optional[torch.Tensor]:
        if self.provider_trainable is None:
            return None
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
