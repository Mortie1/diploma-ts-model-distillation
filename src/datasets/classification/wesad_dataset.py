from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.datasets.classification.cache_utils import (
    cache_signature,
    load_cached_index,
    load_tensor_file,
    materialize_tensor_cache,
)
from src.datasets.download import maybe_download_wesad

logger = logging.getLogger(__name__)


class WESADDataset(BaseDataset):
    """
    WESAD mood classification from wrist BVP (PPG).
    """

    SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 3840,  # 60s @ 64Hz
        stride: int = 1920,
        val_subject: int = 2,
        test_subject: int = 3,
        allowed_labels: tuple[int, ...] = (1, 2, 3, 4),
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
        if val_subject == test_subject:
            raise ValueError("val_subject and test_subject must be different")
        allowed = tuple(int(x) for x in allowed_labels)
        if not allowed:
            raise ValueError("allowed_labels must be non-empty")

        root_path = Path(root)
        if not any(root_path.rglob("S*/S*.pkl")):
            logger.info("WESAD not found under %s, downloading...", root_path)
            maybe_download_wesad(root_path)

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "wesad",
            "split": split_key,
            "window_length": int(window_length),
            "stride": int(stride),
            "val_subject": int(val_subject),
            "test_subject": int(test_subject),
            "allowed_labels": list(allowed),
            "normalize": bool(normalize),
            "root": str(root_path.resolve()),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "wesad" / split_key / signature
        cached = load_cached_index(cache_dir)
        if cached is not None:
            logger.info("WESAD cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array([int(x["label"]) for x in cached], dtype=np.int64)
            super().__init__(index=cached, *args, **kwargs)
            return
        logger.info("WESAD cache miss: split=%s -> %s", split_key, cache_dir)

        train_subjects = set(self.SUBJECTS) - {int(val_subject), int(test_subject)}
        split_subjects = {
            "train": train_subjects,
            "val": {int(val_subject)},
            "test": {int(test_subject)},
        }[split_key]

        label_to_idx = {label: i for i, label in enumerate(sorted(allowed))}
        samples: list[tuple[np.ndarray, int]] = []
        for sid in split_subjects:
            pkl_candidates = [
                root_path / f"S{sid}" / f"S{sid}.pkl",
                root_path / "WESAD" / f"S{sid}" / f"S{sid}.pkl",
            ]
            pkl_path = next((p for p in pkl_candidates if p.exists()), None)
            if pkl_path is None:
                continue
            with pkl_path.open("rb") as f:
                obj = pickle.load(f, encoding="latin1")
            if "signal" not in obj or "label" not in obj:
                continue
            signal = obj["signal"]
            if "wrist" not in signal or "BVP" not in signal["wrist"]:
                continue
            x = np.asarray(signal["wrist"]["BVP"], dtype=np.float32).reshape(-1, 1)
            y = np.asarray(obj["label"], dtype=np.int64).reshape(-1)
            n = min(len(x), len(y))
            x = x[:n]
            y = y[:n]
            mask = np.isin(y, np.asarray(allowed, dtype=np.int64))
            x = x[mask]
            y = y[mask]
            if len(y) < window_length:
                continue
            for start in range(0, len(y) - window_length + 1, stride):
                end = start + window_length
                y_win = y[start:end]
                labels, counts = np.unique(y_win, return_counts=True)
                label_raw = int(labels[int(np.argmax(counts))])
                win = x[start:end].T.astype(np.float32)  # [1,T]
                if normalize:
                    mean = win.mean(axis=1, keepdims=True)
                    std = win.std(axis=1, keepdims=True) + 1e-6
                    win = (win - mean) / std
                samples.append((win, int(label_to_idx[label_raw])))

        if not samples:
            raise RuntimeError("No WESAD windows were generated.")

        self.labels = np.asarray([int(y) for _, y in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(y)) for x, y in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching wesad/{split_key}",
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
