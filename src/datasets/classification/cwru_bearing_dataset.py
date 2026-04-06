from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from src.datasets.base_dataset import BaseDataset
from src.datasets.classification.cache_utils import (
    cache_signature,
    load_cached_index,
    load_tensor_file,
    materialize_tensor_cache,
)

logger = logging.getLogger(__name__)


class CWRUBearingDataset(BaseDataset):
    """
    CWRU bearing classification dataset (HF mirror) with train/val/test split.

    Source: BFDS-Project/Bearing-Fault-Diagnosis-System
    """

    HF_REPO = "BFDS-Project/Bearing-Fault-Diagnosis-System"
    VARIANT_TO_FILE = {
        "12k_drive_end": "CWRU1024/12kDriveEnd.csv",
        "12k_fan_end": "CWRU1024/12kFanEnd.csv",
        "48k_drive_end": "CWRU1024/48kDriveEnd.csv",
    }

    def __init__(
        self,
        root: str,
        split: str,
        variant: str = "12k_drive_end",
        normalize: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        split_seed: int = 42,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")
        variant_key = variant.lower()
        if variant_key not in self.VARIANT_TO_FILE:
            raise ValueError(
                f"Unknown variant `{variant}`. Available: {sorted(self.VARIANT_TO_FILE)}"
            )
        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0, 1)")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1")

        root_path = Path(root)
        csv_path = self._resolve_or_download_csv(root_path, variant_key)

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "cwru_bearing",
            "split": split_key,
            "variant": variant_key,
            "normalize": bool(normalize),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "split_seed": int(split_seed),
            "csv_path": str(csv_path.resolve()),
            "csv_size": csv_path.stat().st_size,
            "csv_mtime_ns": csv_path.stat().st_mtime_ns,
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "cwru_bearing" / variant_key / split_key / signature

        cached_index = load_cached_index(cache_dir)
        if cached_index is not None:
            logger.info(
                "CWRU cache hit: split=%s variant=%s path=%s",
                split_key,
                variant_key,
                cache_dir,
            )
            self.labels = np.array(
                [int(x["label"]) for x in cached_index], dtype=np.int64
            )
            super().__init__(index=cached_index, *args, **kwargs)
            return
        logger.info(
            "CWRU cache miss: split=%s variant=%s -> building cache at %s",
            split_key,
            variant_key,
            cache_dir,
        )

        arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        x = arr[:, :-1].astype(np.float32)
        y_raw = arr[:, -1].astype(np.int64)
        unique_labels = sorted(np.unique(y_raw).tolist())
        label_map = {value: idx for idx, value in enumerate(unique_labels)}
        y = np.array([label_map[v] for v in y_raw], dtype=np.int64)

        split_idx = self._make_stratified_split_indices(
            labels=y,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            seed=int(split_seed),
        )
        indices = split_idx[split_key]
        x = x[indices]
        y = y[indices]

        if normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std

        self.labels = y.astype(np.int64, copy=False)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x[i][None, :], int(y[i])) for i in range(len(y))),
            meta={**cfg, "signature": signature, "n_samples": int(len(y))},
            total=int(len(y)),
            progress_desc=f"caching cwru/{variant_key}/{split_key}",
        )
        super().__init__(index=index, *args, **kwargs)

    @classmethod
    def _resolve_or_download_csv(cls, root_path: Path, variant_key: str) -> Path:
        filename = cls.VARIANT_TO_FILE[variant_key]
        local_path = root_path / "CWRU" / Path(filename).name
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cached_file = hf_hub_download(
            repo_id=cls.HF_REPO,
            repo_type="dataset",
            filename=filename,
        )
        shutil.copyfile(cached_file, local_path)
        return local_path

    @staticmethod
    def _make_stratified_split_indices(
        labels: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            cls_idx = cls_idx.copy()
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_idx.extend(cls_idx[:n_train].tolist())
            val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
            test_idx.extend(cls_idx[n_train + n_val :].tolist())
        return {
            "train": np.array(train_idx, dtype=np.int64),
            "val": np.array(val_idx, dtype=np.int64),
            "test": np.array(test_idx, dtype=np.int64),
        }

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
