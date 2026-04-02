from src.datasets.catalog import (
    ETTh1ForecastDataset,
    ETTh2ForecastDataset,
    ETTm1ForecastDataset,
    ETTm2ForecastDataset,
    ForecastCSVWindowDataset,
    LTSFElectricityForecastDataset,
    LTSFTrafficForecastDataset,
    LTSFWeatherForecastDataset,
    UCRDataset,
)
from src.datasets.example import ExampleDataset
from src.datasets.ts_classification import SyntheticClassificationDataset
from src.datasets.ts_forecasting import SyntheticForecastingDataset

__all__ = [
    "ExampleDataset",
    "SyntheticClassificationDataset",
    "SyntheticForecastingDataset",
    "UCRDataset",
    "ForecastCSVWindowDataset",
    "ETTh1ForecastDataset",
    "ETTh2ForecastDataset",
    "ETTm1ForecastDataset",
    "ETTm2ForecastDataset",
    "LTSFElectricityForecastDataset",
    "LTSFTrafficForecastDataset",
    "LTSFWeatherForecastDataset",
]
