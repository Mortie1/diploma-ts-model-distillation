from __future__ import annotations

import math

import torch

from src.datasets.base_dataset import BaseDataset


class SmokeForecastingDataset(BaseDataset):
    """Synthetic TS forecasting dataset (context -> horizon)."""

    def __init__(
        self,
        n_samples: int,
        context_length: int,
        horizon: int,
        n_channels: int = 1,
        *args,
        **kwargs,
    ):
        self.context_length = context_length
        self.horizon = horizon
        self.total_length = context_length + horizon
        self.n_channels = n_channels
        index = [{"path": str(i), "label": 0} for i in range(n_samples)]
        super().__init__(index=index, *args, **kwargs)

    def load_object(self, path):
        idx = int(path)
        t = torch.linspace(0, 1, steps=self.total_length)
        phase = (idx % 10) / 10.0
        trend = 0.3 * t
        signal = torch.sin(2 * math.pi * (2 + phase) * t) + trend
        signal = signal + 0.05 * torch.randn_like(signal)
        return signal.unsqueeze(0).repeat(self.n_channels, 1)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        series = self.load_object(data_dict["path"])
        inputs = series[:, : self.context_length]
        targets = series[:, self.context_length :]
        instance_data = {
            "inputs": inputs,
            "targets": targets,
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data


class SyntheticForecastingDataset(SmokeForecastingDataset):
    """Backward-compatible alias."""
