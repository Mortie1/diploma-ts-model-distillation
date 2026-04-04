from src.datasets.classification import PAMAP2Dataset, UCRDataset
from src.datasets.example import ExampleDataset
from src.datasets.smoke_classification import (
    SmokeClassificationDataset,
    SyntheticClassificationDataset,
)

__all__ = [
    "ExampleDataset",
    "PAMAP2Dataset",
    "SmokeClassificationDataset",
    "SyntheticClassificationDataset",
    "UCRDataset",
]
