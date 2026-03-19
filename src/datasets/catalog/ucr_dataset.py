from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.catalog.download import maybe_download_ucr


class UCRDataset(BaseDataset):
    """Loader for UCR archive TSV files (first column = label)."""

    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str,
        normalize: bool = True,
        *args,
        **kwargs,
    ):
        root_path = Path(root)
        split_name = "TRAIN" if split.lower() == "train" else "TEST"
        candidates = [
            root_path / dataset_name / f"{dataset_name}_{split_name}.tsv",
            root_path / dataset_name / f"{dataset_name}_{split_name}.txt",
            root_path / f"{dataset_name}_{split_name}.tsv",
            root_path / f"{dataset_name}_{split_name}.txt",
        ]
        file_path = next((p for p in candidates if p.exists()), None)
        if file_path is None:
            maybe_download_ucr(dataset_name=dataset_name, root=root_path)
            file_path = next((p for p in candidates if p.exists()), None)
            if file_path is None:
                checked = ", ".join(str(p) for p in candidates)
                raise FileNotFoundError(f"Could not find UCR split file. Checked: {checked}")

        # Many UCR files are tab-separated; some exports are plain whitespace text.
        arr = np.loadtxt(file_path, delimiter=None)

        labels = arr[:, 0].astype(np.int64)
        unique_labels = sorted(np.unique(labels).tolist())
        label_map = {value: idx for idx, value in enumerate(unique_labels)}

        self.samples = arr[:, 1:].astype(np.float32)
        if normalize:
            mean = self.samples.mean(axis=1, keepdims=True)
            std = self.samples.std(axis=1, keepdims=True) + 1e-6
            self.samples = (self.samples - mean) / std
        self.labels = np.array([label_map[val] for val in labels], dtype=np.int64)

        index = [{"path": str(i), "label": int(self.labels[i])} for i in range(len(self.labels))]
        super().__init__(index=index, *args, **kwargs)

    def load_object(self, path):
        i = int(path)
        x = torch.from_numpy(self.samples[i]).unsqueeze(0)
        return x

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        x = self.load_object(data_dict["path"])
        y = int(data_dict["label"])
        instance_data = {"x": x, "y": y}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
