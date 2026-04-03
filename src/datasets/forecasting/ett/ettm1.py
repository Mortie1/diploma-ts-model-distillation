from __future__ import annotations

from src.datasets.forecasting.ett.constants import ETT_DEFAULT_COLUMNS
from src.datasets.forecasting.window_dataset import ForecastCSVWindowDataset


class ETTm1ForecastDataset(ForecastCSVWindowDataset):
    def __init__(
        self,
        context_length: int,
        horizon: int,
        split: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        csv_path: str = "data/raw/ett/ETTm1.csv",
        value_columns: list[str] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            csv_path=csv_path,
            value_columns=value_columns or ETT_DEFAULT_COLUMNS,
            context_length=context_length,
            horizon=horizon,
            split=split,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            *args,
            **kwargs,
        )
