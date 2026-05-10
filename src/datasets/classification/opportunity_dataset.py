from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.datasets.base_dataset import BaseDataset
from src.datasets.classification.cache_utils import (
    cache_signature,
    load_cached_index,
    load_tensor_file,
    materialize_tensor_cache,
)
from src.datasets.download import maybe_download_opportunity

logger = logging.getLogger(__name__)


class OpportunityDataset(BaseDataset):
    """
    Opportunity dataset with locomotion protocol (4 classes) and wrist accel channels.
    """

    # 1-based raw columns in UCI docs: accelerometer RWR x/y/z = 22/23/24.
    WRIST_ACCEL_COLS_0BASED = [21, 22, 23]
    # Locomotion label is at 0-indexed column 243 (1-indexed 244) in the UCI format.
    # Column 242 is the last sensor column, not a label.
    LOCOMOTION_LABEL_CANDIDATES = [243, 242]
    LOCOMOTION_MAP = {1: 0, 2: 1, 4: 2, 5: 3}  # stand, walk, sit, lie
    SUBJECTS = [1, 2, 3, 4]

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 1000,
        stride: int = 500,
        target_hz: int = 100,
        raw_hz: int = 30,
        val_subject: int = 1,
        test_subject: int = 2,
        min_majority_ratio: float = 0.5,
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
        if target_hz <= 0 or raw_hz <= 0:
            raise ValueError("target_hz and raw_hz must be > 0")
        if val_subject == test_subject:
            raise ValueError("val_subject and test_subject must be different")

        root_path = Path(root)
        data_dir = root_path / "OpportunityUCIDataset" / "dataset"
        if not data_dir.exists():
            logger.info("Opportunity not found at %s, downloading...", data_dir)
            maybe_download_opportunity(root_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Opportunity directory not found: {data_dir}")

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "opportunity",
            "schema_version": 2,
            "split": split_key,
            "window_length": int(window_length),
            "stride": int(stride),
            "target_hz": int(target_hz),
            "raw_hz": int(raw_hz),
            "val_subject": int(val_subject),
            "test_subject": int(test_subject),
            "min_majority_ratio": float(min_majority_ratio),
            "normalize": bool(normalize),
            "root": str(root_path.resolve()),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "opportunity" / split_key / signature
        cached = load_cached_index(cache_dir)
        if cached is not None:
            logger.info("Opportunity cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array([int(x["label"]) for x in cached], dtype=np.int64)
            super().__init__(index=cached, *args, **kwargs)
            return
        logger.info("Opportunity cache miss: split=%s -> %s", split_key, cache_dir)

        train_subjects = set(self.SUBJECTS) - {int(val_subject), int(test_subject)}
        split_subjects = {
            "train": train_subjects,
            "val": {int(val_subject)},
            "test": {int(test_subject)},
        }[split_key]

        samples: list[tuple[np.ndarray, int]] = []
        for subj in split_subjects:
            for file_path in sorted(data_dir.glob(f"S{subj}-*.dat")):
                try:
                    arr = np.loadtxt(file_path, dtype=np.float32)
                except Exception:
                    continue
                if arr.ndim != 2:
                    continue
                if arr.shape[1] <= max(self.WRIST_ACCEL_COLS_0BASED):
                    continue
                label_col = None
                for cand in self.LOCOMOTION_LABEL_CANDIDATES:
                    if cand < arr.shape[1]:
                        label_col = cand
                        break
                if label_col is None:
                    continue
                x = arr[:, self.WRIST_ACCEL_COLS_0BASED].astype(np.float32)  # [T,3]
                y_raw_f = arr[:, label_col].astype(np.float32)
                finite_label_mask = np.isfinite(y_raw_f)
                y_raw = np.round(y_raw_f[finite_label_mask]).astype(np.int64)
                x = x[finite_label_mask]

                # Clean invalid sensor values before label filtering.
                x = self._fill_nan_columns(x)

                keep_mask = np.isin(y_raw, np.array(list(self.LOCOMOTION_MAP.keys())))
                if keep_mask.sum() < 10:
                    continue
                x = x[keep_mask]
                y = np.array([self.LOCOMOTION_MAP[int(v)] for v in y_raw[keep_mask]])

                # Resample to target_hz per-axis to match article protocol.
                if target_hz != raw_hz:
                    x = self._resample_series(x, src_hz=raw_hz, dst_hz=target_hz)
                    y = self._resample_labels(y, src_hz=raw_hz, dst_hz=target_hz)
                n = len(y)
                if n < window_length:
                    continue
                for start in range(0, n - window_length + 1, stride):
                    end = start + window_length
                    y_win = y[start:end]
                    labels, counts = np.unique(y_win, return_counts=True)
                    best = int(np.argmax(counts))
                    maj = int(labels[best])
                    ratio = float(counts[best]) / float(window_length)
                    if ratio < min_majority_ratio:
                        continue
                    win = x[start:end].T.astype(np.float32)  # [3,T]
                    if normalize:
                        mean = win.mean(axis=1, keepdims=True)
                        std = win.std(axis=1, keepdims=True) + 1e-6
                        win = (win - mean) / std
                    samples.append((win, maj))

        if not samples:
            raise RuntimeError("No Opportunity windows were generated.")

        self.labels = np.asarray([int(y) for _, y in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(y)) for x, y in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching opportunity/{split_key}",
        )
        super().__init__(index=index, *args, **kwargs)

    @staticmethod
    def _resample_series(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
        n = x.shape[0]
        t_src = np.arange(n, dtype=np.float32) / float(src_hz)
        n_dst = max(1, int(round(n * float(dst_hz) / float(src_hz))))
        t_dst = np.arange(n_dst, dtype=np.float32) / float(dst_hz)
        out = np.zeros((n_dst, x.shape[1]), dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = np.interp(t_dst, t_src, x[:, c]).astype(np.float32)
        return out

    @staticmethod
    def _resample_labels(y: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
        n = y.shape[0]
        n_dst = max(1, int(round(n * float(dst_hz) / float(src_hz))))
        idx = np.clip(
            np.round(np.linspace(0, n - 1, n_dst)).astype(np.int64),
            0,
            n - 1,
        )
        return y[idx]

    @staticmethod
    def _fill_nan_columns(features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return features.astype(np.float32)
        x = features.astype(np.float32, copy=True)
        x[~np.isfinite(x)] = np.nan
        if not np.isnan(x).any():
            return x
        col_median = np.nanmedian(x, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median).astype(np.float32)
        nan_rows, nan_cols = np.where(np.isnan(x))
        x[nan_rows, nan_cols] = col_median[nan_cols]
        return x

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
