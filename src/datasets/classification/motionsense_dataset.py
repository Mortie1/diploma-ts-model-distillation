from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.datasets.classification.cache_utils import (
    cache_signature,
    load_cached_index,
    load_tensor_file,
    materialize_tensor_cache,
)
from src.datasets.download import maybe_download_motionsense

logger = logging.getLogger(__name__)


class MotionSenseDataset(BaseDataset):
    """
    MotionSense HAR dataset (iPhone inertial, 50Hz, 6 classes).

    Expected raw layout: <root>/A_DeviceMotion_data/<act>_<trial>/sub_<id>.csv
    """

    TRIAL_CODES = {
        "dws": [1, 2, 11],
        "ups": [3, 4, 12],
        "sit": [5, 13],
        "std": [6, 14],
        "wlk": [7, 8, 15],
        "jog": [9, 16],
    }
    LABEL_TO_IDX = {"dws": 0, "ups": 1, "sit": 2, "std": 3, "wlk": 4, "jog": 5}
    SUBJECTS = list(range(1, 25))

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 200,
        stride: int = 100,
        upsample_factor: int = 1,
        fold_index: int = 0,
        n_folds: int = 5,
        normalize: bool = True,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")
        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be > 0")
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0")
        if n_folds <= 1:
            raise ValueError("n_folds must be > 1")
        if fold_index < 0 or fold_index >= n_folds:
            raise ValueError("fold_index must be in [0, n_folds)")

        root_path = Path(root)
        data_dir = root_path / "A_DeviceMotion_data"
        if not data_dir.exists():
            logger.info("MotionSense not found at %s, downloading...", data_dir)
            maybe_download_motionsense(root_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"MotionSense directory not found: {data_dir}")

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "motionsense",
            "split": split_key,
            "window_length": int(window_length),
            "stride": int(stride),
            "upsample_factor": int(upsample_factor),
            "fold_index": int(fold_index),
            "n_folds": int(n_folds),
            "normalize": bool(normalize),
            "root": str(root_path.resolve()),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "motionsense" / split_key / signature
        cached = load_cached_index(cache_dir)
        if cached is not None:
            logger.info("MotionSense cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array([int(x["label"]) for x in cached], dtype=np.int64)
            super().__init__(index=cached, *args, **kwargs)
            return
        logger.info(
            "MotionSense cache miss: split=%s -> building cache at %s",
            split_key,
            cache_dir,
        )

        fold_subjects = np.array_split(np.array(self.SUBJECTS, dtype=np.int64), n_folds)
        test_subjects = set(fold_subjects[fold_index].tolist())
        val_subjects = set(fold_subjects[(fold_index + 1) % n_folds].tolist())
        train_subjects = set(self.SUBJECTS) - test_subjects - val_subjects
        split_subjects = {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        }[split_key]

        samples: list[tuple[np.ndarray, int]] = []
        for act, trials in self.TRIAL_CODES.items():
            label = self.LABEL_TO_IDX[act]
            for trial in trials:
                trial_dir = data_dir / f"{act}_{trial}"
                if not trial_dir.exists():
                    continue
                for sub in split_subjects:
                    csv_path = trial_dir / f"sub_{sub}.csv"
                    if not csv_path.exists():
                        continue
                    arr = pd.read_csv(csv_path)
                    if "Unnamed: 0" in arr.columns:
                        arr = arr.drop(columns=["Unnamed: 0"])
                    x = arr.to_numpy(dtype=np.float32)  # [T, C]
                    if upsample_factor > 1:
                        x = self._upsample_series(x, factor=upsample_factor)
                    if x.ndim != 2 or x.shape[0] < window_length:
                        continue
                    for start in range(0, x.shape[0] - window_length + 1, stride):
                        win = x[start : start + window_length].T.astype(
                            np.float32
                        )  # [C,T]
                        if normalize:
                            mean = win.mean(axis=1, keepdims=True)
                            std = win.std(axis=1, keepdims=True) + 1e-6
                            win = (win - mean) / std
                        samples.append((win, int(label)))

        if not samples:
            raise RuntimeError("No MotionSense samples were generated.")

        self.labels = np.asarray([int(y) for _, y in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(y)) for x, y in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching motionsense/{split_key}",
        )
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

    @staticmethod
    def _upsample_series(x: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return x
        t = x.shape[0]
        if t <= 1:
            return x
        t_new = t * factor
        src = np.arange(t, dtype=np.float32)
        dst = np.linspace(0.0, float(t - 1), num=t_new, dtype=np.float32)
        out = np.empty((t_new, x.shape[1]), dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = np.interp(dst, src, x[:, c]).astype(np.float32)
        return out
