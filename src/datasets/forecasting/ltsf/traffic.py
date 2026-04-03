from __future__ import annotations

from src.datasets.forecasting.window_dataset import ForecastCSVWindowDataset


class LTSFTrafficForecastDataset(ForecastCSVWindowDataset):
    def __init__(
        self,
        context_length: int,
        horizon: int,
        split: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        normalize_with_train_stats: bool = True,
        csv_path: str = "data/raw/ltsf/traffic.csv",
        *args,
        **kwargs,
    ):
        super().__init__(
            csv_path=csv_path,
            value_columns=None,
            context_length=context_length,
            horizon=horizon,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            normalize_with_train_stats=normalize_with_train_stats,
            *args,
            **kwargs,
        )
