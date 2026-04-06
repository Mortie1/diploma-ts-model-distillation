from __future__ import annotations

from src.datasets.classification.ucr_dataset import UCRDataset


class InsectWingbeatDataset(UCRDataset):
    """Convenience wrapper around UCR InsectWingbeat dataset."""

    def __init__(
        self,
        root: str,
        split: str,
        dataset_name: str = "InsectWingbeatSound",
        normalize: bool = True,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            root=root,
            dataset_name=dataset_name,
            split=split,
            normalize=normalize,
            cache_root=cache_root,
            *args,
            **kwargs,
        )
