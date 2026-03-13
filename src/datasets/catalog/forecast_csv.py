from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.datasets.base_dataset import BaseDataset


class ForecastCSVWindowDataset(BaseDataset):
    """Sliding-window dataset for forecasting from multivariate CSV."""

    def __init__(
        self,
        csv_path: str,
        value_columns: list[str],
        context_length: int,
        horizon: int,
        split: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        normalize_with_train_stats: bool = True,
        *args,
        **kwargs,
    ):
        self.context_length = context_length
        self.horizon = horizon

        df = pd.read_csv(Path(csv_path))
        values = df[value_columns].to_numpy(dtype=np.float32)

        n = len(values)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if split == "train":
            split_values = values[:train_end]
            offset = 0
        elif split == "val":
            split_values = values[train_end:val_end]
            offset = train_end
        elif split == "test":
            split_values = values[val_end:]
            offset = val_end
        else:
            raise ValueError(f"Unknown split: {split}")

        if normalize_with_train_stats:
            train_vals = values[:train_end]
            mean = train_vals.mean(axis=0, keepdims=True)
            std = train_vals.std(axis=0, keepdims=True) + 1e-6
            split_values = (split_values - mean) / std

        self.values = split_values
        max_start = len(self.values) - (context_length + horizon)
        index = []
        for start in range(max_start):
            index.append({"path": str(start), "label": 0, "offset": offset})

        super().__init__(index=index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        start = int(data_dict["path"])
        end_ctx = start + self.context_length
        end_tgt = end_ctx + self.horizon

        chunk = self.values[start:end_tgt]
        context = torch.from_numpy(chunk[: self.context_length]).transpose(0, 1)
        target = torch.from_numpy(chunk[self.context_length :]).transpose(0, 1)

        instance_data = {
            "context": context,
            "target": target,
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data
