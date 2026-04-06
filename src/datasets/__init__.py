from src.datasets.classification import (
    CWRUBearingDataset,
    InsectWingbeatDataset,
    PAMAP2Dataset,
    PTBXLDataset,
    UCRDataset,
)
from src.datasets.example import ExampleDataset
from src.datasets.smoke_classification import (
    SmokeClassificationDataset,
    SyntheticClassificationDataset,
)

__all__ = [
    "ExampleDataset",
    "CWRUBearingDataset",
    "InsectWingbeatDataset",
    "PAMAP2Dataset",
    "PTBXLDataset",
    "SmokeClassificationDataset",
    "SyntheticClassificationDataset",
    "UCRDataset",
]
