from src.loss.example import ExampleLoss
from src.loss.classification import ClassificationLoss, DistillClassificationLoss
from src.loss.forecasting import DistillForecastingLoss, ForecastingLoss

__all__ = [
    "ExampleLoss",
    "ClassificationLoss",
    "DistillClassificationLoss",
    "ForecastingLoss",
    "DistillForecastingLoss",
]
