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
from src.datasets.download import maybe_download_pamap2

logger = logging.getLogger(__name__)


class PAMAP2Dataset(BaseDataset):
    """
    PAMAP2 HAR dataset with fixed-size sliding windows.

    Returns:
        inputs: Tensor [C, T]
        targets: int class id
    """

    # Activity IDs present in PAMAP2 Protocol files used by this project.
    # We intentionally keep labels compact (0..N-1) for classification heads.
    _ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    #  1	lying
    #  2	sitting
    #  3	standing
    #  4	walking
    #  5	running
    #  6	cycling
    #  7	Nordic walking
    #  9	watching TV
    #  10	computer work
    #  11	car driving
    #  12	ascending stairs
    #  13	descending stairs
    #  16	vacuum cleaning
    #  17	ironing
    #  18	folding laundry
    #  19	house cleaning
    #  20	playing soccer
    #  24	rope jumping
    _ACTIVITY_TO_INDEX = {aid: idx for idx, aid in enumerate(_ACTIVITY_IDS)}
    _SPLIT_SUBJECTS = {
        "train": [101, 102, 103, 104, 105, 106],
        "val": [107],
        "test": [108, 109],
    }

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 256,
        stride: int = 128,
        feature_set: str = "all",
        include_hr: bool = True,
        min_majority_ratio: float = 0.6,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        if split_key not in self._SPLIT_SUBJECTS:
            raise ValueError(
                f"Unknown split `{split}`. Expected one of: train/val/test."
            )
        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be positive.")

        root_path = Path(root)
        protocol_dir = root_path / "PAMAP2_Dataset" / "Protocol"
        if not protocol_dir.exists():
            logger.info("PAMAP2 not found at %s, downloading...", protocol_dir)
            maybe_download_pamap2(root=root_path)
        if not protocol_dir.exists():
            raise FileNotFoundError(
                f"PAMAP2 protocol directory not found: {protocol_dir}"
            )

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "pamap2",
            "split": split_key,
            "window_length": int(window_length),
            "stride": int(stride),
            "feature_set": str(feature_set),
            "include_hr": bool(include_hr),
            "min_majority_ratio": float(min_majority_ratio),
            "root": str(root_path.resolve()),
            "label_schema_version": 2,
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "pamap2" / split_key / signature
        cached_index = load_cached_index(cache_dir)
        if cached_index is not None:
            logger.info("PAMAP2 cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array(
                [int(x["label"]) for x in cached_index], dtype=np.int64
            )
            super().__init__(index=cached_index, *args, **kwargs)
            return
        logger.info(
            "PAMAP2 cache miss: split=%s -> building cache at %s", split_key, cache_dir
        )

        samples: list[tuple[np.ndarray, int]] = []

        for subject_id in self._SPLIT_SUBJECTS[split_key]:
            file_path = protocol_dir / f"subject{subject_id}.dat"
            if not file_path.exists():
                continue

            arr = np.loadtxt(file_path, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue

            activity = arr[:, 1].astype(np.int64)
            features = arr[:, 2:].astype(np.float32)
            features = self._select_features(
                features=features,
                feature_set=feature_set,
                include_hr=include_hr,
            )
            features = self._fill_nan_columns(features)

            valid_mask = activity > 0
            activity = activity[valid_mask]
            features = features[valid_mask]

            n = len(activity)
            if n < window_length:
                continue

            for start in range(0, n - window_length + 1, stride):
                end = start + window_length
                y_window = activity[start:end]
                label_raw, count = np.unique(y_window, return_counts=True)
                best_idx = int(np.argmax(count))
                majority_label = int(label_raw[best_idx])
                majority_ratio = float(count[best_idx]) / float(window_length)
                if majority_ratio < min_majority_ratio:
                    continue
                if majority_label not in self._ACTIVITY_TO_INDEX:
                    continue

                x = features[start:end].T  # [C, T]

                samples.append(
                    (x.astype(np.float32), self._ACTIVITY_TO_INDEX[majority_label])
                )

        if not samples:
            raise RuntimeError(
                "No PAMAP2 windows were generated. "
                "Check split, window_length/stride, and dataset files."
            )

        self.labels = np.asarray([int(label) for _, label in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(label)) for x, label in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching pamap2/{split_key}",
        )
        super().__init__(index=index, *args, **kwargs)

    @staticmethod
    def _fill_nan_columns(features: np.ndarray) -> np.ndarray:
        if not np.isnan(features).any():
            return features
        col_median = np.nanmedian(features, axis=0)
        col_median = np.where(np.isnan(col_median), 0.0, col_median).astype(np.float32)
        nan_rows, nan_cols = np.where(np.isnan(features))
        features = features.copy()
        features[nan_rows, nan_cols] = col_median[nan_cols]
        return features

    @staticmethod
    def _select_features(
        features: np.ndarray, feature_set: str, include_hr: bool
    ) -> np.ndarray:
        fs = feature_set.lower()
        n_feat = features.shape[1]

        # Features after removing timestamp/activity:
        # [0]=heart_rate, then 3 IMU blocks (17 each): hand/chest/ankle.
        if fs in {"hr", "heart_rate", "heartbeat"}:
            return features[:, [0]]

        if fs == "all":
            return features

        if fs == "acc_gyro" and n_feat >= 52:
            idx: list[int] = []
            if include_hr:
                idx.append(0)
            for base in (1, 18, 35):
                idx.extend(range(base + 4, base + 10))  # acc6 xyz + gyro xyz
            return features[:, idx]

        return features

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
