from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.datasets.classification.cache_utils import (
    cache_signature,
    load_cached_index,
    load_tensor_file,
    materialize_tensor_cache,
)
from src.datasets.download import maybe_download_ucr


class UCRDataset(BaseDataset):
    """Loader for UCR archive TSV files (first column = label)."""

    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str,
        normalize: bool = True,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        root_path = Path(root)
        split_key = split.lower()
        split_name = "TRAIN" if split_key == "train" else "TEST"
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

        cache_base = Path(cache_root) if cache_root else (root_path / ".cache" / "classification")
        cfg = {
            "dataset": "ucr",
            "dataset_name": dataset_name,
            "split": split_key,
            "normalize": bool(normalize),
            "file_path": str(file_path.resolve()),
            "file_size": file_path.stat().st_size,
            "file_mtime_ns": file_path.stat().st_mtime_ns,
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "ucr" / dataset_name / split_key / signature

        cached_index = load_cached_index(cache_dir)
        if cached_index is not None:
            self.labels = np.array([int(x["label"]) for x in cached_index], dtype=np.int64)
            super().__init__(index=cached_index, *args, **kwargs)
            return

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

        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((self.samples[i][None, :], int(self.labels[i])) for i in range(len(self.labels))),
            meta={**cfg, "signature": signature, "n_samples": int(len(self.labels))},
        )
        self.samples = None  # no RAM copy after cache materialization
        super().__init__(index=index, *args, **kwargs)

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
