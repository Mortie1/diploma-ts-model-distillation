from src.datasets.example import ExampleDataset
from src.datasets.ts_classification import SyntheticClassificationDataset
from src.datasets.ts_forecasting import SyntheticForecastingDataset
from src.datasets.catalog.forecast_csv import ForecastCSVWindowDataset
from src.datasets.catalog.ucr_dataset import UCRDataset

__all__ = [
    "ExampleDataset",
    "SyntheticClassificationDataset",
    "SyntheticForecastingDataset",
    "UCRDataset",
    "ForecastCSVWindowDataset",
]
