from __future__ import annotations

import math

import torch

from src.datasets.base_dataset import BaseDataset


class SmokeClassificationDataset(BaseDataset):
    """Synthetic TS classification dataset used for smoke/e2e runs."""

    def __init__(
        self,
        n_samples: int,
        length: int,
        n_classes: int,
        n_channels: int = 1,
        name: str = "train",
        *args,
        **kwargs,
    ):
        self.length = length
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.name = name

        index = [{"path": str(i), "label": i % n_classes} for i in range(n_samples)]
        super().__init__(index=index, *args, **kwargs)

    def load_object(self, path):
        idx = int(path)
        label = idx % self.n_classes

        t = torch.linspace(0, 1, steps=self.length)
        freq = 1 + label
        base = torch.sin(2 * math.pi * freq * t)
        noise = 0.05 * torch.randn(self.length)
        x = (base + noise).unsqueeze(0).repeat(self.n_channels, 1)
        return x

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        x = self.load_object(data_dict["path"])
        y = int(data_dict["label"])
        instance_data = {"x": x, "y": y}
        instance_data = self.preprocess_data(instance_data)
        return instance_data


class SyntheticClassificationDataset(SmokeClassificationDataset):
    """Backward-compatible alias."""
