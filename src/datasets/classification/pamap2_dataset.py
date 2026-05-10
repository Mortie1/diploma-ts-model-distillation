from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

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

    # Default protocol used in this project (12 classes).
    _ACTIVITY_IDS_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    _SPLIT_SUBJECTS_DEFAULT = {
        "train": [101, 102, 103, 104, 105, 106],
        "val": [107],
        "test": [108, 109],
    }

    # ssl-wearables-compatible protocol (8 classes, subject109 removed).
    _ACTIVITY_IDS_SSL = [1, 2, 3, 4, 12, 13, 16, 17]
    _SPLIT_SUBJECTS_SSL = {
        "train": [101, 102, 103, 104, 105, 106],
        "val": [107],
        "test": [108],
    }
    # Article-like 2s/100Hz setup (12 classes, 5-fold style can be emulated via subject overrides).
    _ACTIVITY_IDS_LEG_12 = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
    _SPLIT_SUBJECTS_LEG_12 = {
        "train": [101, 102, 103, 104, 105, 106],
        "val": [107],
        "test": [108],
    }

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 256,
        stride: int = 128,
        upsample_factor: int = 1,
        feature_set: str = "all",
        imu_sensors: str = "all",
        protocol_mode: str = "default",
        train_subjects: list[int] | None = None,
        val_subjects: list[int] | None = None,
        test_subjects: list[int] | None = None,
        use_optional: bool = False,
        include_hr: bool = True,
        min_majority_ratio: float = 0.6,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        protocol_key = str(protocol_mode).lower()
        activity_ids, split_subjects = self._resolve_protocol(
            protocol_mode=protocol_key,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )
        if split_key not in split_subjects:
            raise ValueError(
                f"Unknown split `{split}`. Expected one of: train/val/test."
            )
        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be positive.")
        if upsample_factor <= 0:
            raise ValueError("upsample_factor must be > 0.")
        self._activity_to_index = {aid: idx for idx, aid in enumerate(activity_ids)}

        root_path = Path(root)
        protocol_dir = root_path / "PAMAP2_Dataset" / "Protocol"
        optional_dir = root_path / "PAMAP2_Dataset" / "Optional"
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
            "upsample_factor": int(upsample_factor),
            "feature_set": str(feature_set),
            "imu_sensors": str(imu_sensors),
            "protocol_mode": protocol_key,
            "train_subjects": (
                list(map(int, train_subjects)) if train_subjects is not None else None
            ),
            "val_subjects": (
                list(map(int, val_subjects)) if val_subjects is not None else None
            ),
            "test_subjects": (
                list(map(int, test_subjects)) if test_subjects is not None else None
            ),
            "use_optional": bool(use_optional),
            "include_hr": bool(include_hr),
            "min_majority_ratio": float(min_majority_ratio),
            "root": str(root_path.resolve()),
            "label_schema_version": 6,
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

        for subject_id in split_subjects[split_key]:
            candidate_files = [protocol_dir / f"subject{subject_id}.dat"]
            if use_optional:
                candidate_files.append(optional_dir / f"subject{subject_id}.dat")

            for file_path in candidate_files:
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
                    imu_sensors=imu_sensors,
                )
                features = self._fill_nan_columns(features)
                if upsample_factor > 1:
                    features = self._upsample_series(features, factor=upsample_factor)
                    activity = self._upsample_labels(activity, factor=upsample_factor)

                n = len(activity)
                if n < window_length:
                    continue

                for start in range(0, n - window_length + 1, stride):
                    end = start + window_length
                    y_window = activity[start:end]
                    # Article-style filtering: drop windows where transition label=0
                    # occupies more than 50% of samples.
                    zero_count = int((y_window == 0).sum())
                    if zero_count > 0.5 * window_length:
                        continue
                    label_raw, count = np.unique(y_window, return_counts=True)
                    best_idx = int(np.argmax(count))
                    majority_label = int(label_raw[best_idx])
                    majority_ratio = float(count[best_idx]) / float(window_length)
                    if majority_ratio < min_majority_ratio:
                        continue
                    if majority_label not in self._activity_to_index:
                        continue

                    x = features[start:end].T  # [C, T]

                    samples.append(
                        (x.astype(np.float32), self._activity_to_index[majority_label])
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

    @staticmethod
    def _upsample_labels(y: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return y
        t = y.shape[0]
        if t <= 1:
            return y
        t_new = t * factor
        dst = np.linspace(0.0, float(t - 1), num=t_new, dtype=np.float32)
        idx = np.clip(np.round(dst).astype(np.int64), 0, t - 1)
        return y[idx]

    @staticmethod
    def _select_features(
        features: np.ndarray,
        feature_set: str,
        include_hr: bool,
        imu_sensors: str = "all",
    ) -> np.ndarray:
        fs = feature_set.lower()
        n_feat = features.shape[1]

        # Features after removing timestamp/activity:
        # [0]=heart_rate, then 3 IMU blocks (17 each): hand/chest/ankle.
        if fs in {"hr", "heart_rate", "heartbeat"}:
            return features[:, [0]]

        if fs == "all":
            return features

        # Single-location subsets from PAMAP2 IMU blocks:
        # hand(wrist): columns [1:18], chest: [18:35], ankle: [35:52]
        if fs in {"wrist", "hand", "chest", "ankle"} and n_feat >= 52:
            base_by_name = {
                "wrist": 1,
                "hand": 1,
                "chest": 18,
                "ankle": 35,
            }
            base = base_by_name[fs]
            idx: list[int] = []
            if include_hr:
                idx.append(0)
            idx.extend(
                PAMAP2Dataset._resolve_location_sensor_indices(base, imu_sensors)
            )
            return features[:, idx]

        if fs == "acc_gyro" and n_feat >= 52:
            idx: list[int] = []
            if include_hr:
                idx.append(0)
            for base in (1, 18, 35):
                idx.extend(range(base + 4, base + 10))  # acc6 xyz + gyro xyz
            return features[:, idx]

        return features

    @staticmethod
    def _resolve_location_sensor_indices(base: int, imu_sensors: str) -> list[int]:
        token_ranges = {
            "temp": [0],
            "temperature": [0],
            "acc16": [1, 2, 3],
            "acc6": [4, 5, 6],
            "gyro": [7, 8, 9],
            "mag": [10, 11, 12],
            "magnetometer": [10, 11, 12],
            "orientation": [13, 14, 15, 16],
            "quat": [13, 14, 15, 16],
        }

        spec = str(imu_sensors).lower().strip()
        if spec in {"", "all", "*"}:
            rel = list(range(17))
            return [base + i for i in rel]

        # Accept "acc_gyro", "acc+gyro", "acc,gyro", etc.
        for sep in ("+", ",", "|"):
            spec = spec.replace(sep, "_")
        parts = [p for p in spec.split("_") if p]

        expanded_parts: list[str] = []
        for p in parts:
            if p == "acc":
                expanded_parts.extend(["acc16", "acc6"])
            elif p == "accgyro":
                expanded_parts.extend(["acc16", "acc6", "gyro"])
            else:
                expanded_parts.append(p)

        rel: list[int] = []
        for p in expanded_parts:
            if p not in token_ranges:
                raise ValueError(
                    f"Unknown imu_sensors token `{p}`. "
                    "Allowed: all,temp,acc16,acc6,acc,gyro,mag,magnetometer,orientation."
                )
            rel.extend(token_ranges[p])

        # keep order, drop duplicates
        rel_unique = list(dict.fromkeys(rel))
        return [base + i for i in rel_unique]

    @classmethod
    def _resolve_protocol(
        cls,
        protocol_mode: str,
        train_subjects: list[int] | None = None,
        val_subjects: list[int] | None = None,
        test_subjects: list[int] | None = None,
    ):
        if protocol_mode == "default":
            activity_ids = cls._ACTIVITY_IDS_DEFAULT
            split_subjects = {
                k: list(v) for k, v in cls._SPLIT_SUBJECTS_DEFAULT.items()
            }
        elif protocol_mode in {"ssl", "ssl_wearables", "yuan24_wrist10s"}:
            activity_ids = cls._ACTIVITY_IDS_SSL
            split_subjects = {k: list(v) for k, v in cls._SPLIT_SUBJECTS_SSL.items()}
        elif protocol_mode in {"hare22_leg2s", "leg12_2s"}:
            activity_ids = cls._ACTIVITY_IDS_LEG_12
            split_subjects = {k: list(v) for k, v in cls._SPLIT_SUBJECTS_LEG_12.items()}
        else:
            raise ValueError(
                f"Unknown protocol_mode `{protocol_mode}`. "
                "Use one of: default, ssl_wearables."
            )

        custom = {
            "train": cls._normalize_subjects(train_subjects),
            "val": cls._normalize_subjects(val_subjects),
            "test": cls._normalize_subjects(test_subjects),
        }
        if any(v is not None for v in custom.values()):
            if custom["val"] is not None:
                split_subjects["val"] = custom["val"]
            if custom["test"] is not None:
                split_subjects["test"] = custom["test"]

            if custom["train"] is not None:
                split_subjects["train"] = custom["train"]
            else:
                # Derive train from the protocol's own subject universe so that subjects
                # outside the protocol (e.g. subject 109 in ssl/leg protocols) are excluded.
                if protocol_mode in {"ssl", "ssl_wearables", "yuan24_wrist10s"}:
                    proto_splits = cls._SPLIT_SUBJECTS_SSL
                elif protocol_mode in {"hare22_leg2s", "leg12_2s"}:
                    proto_splits = cls._SPLIT_SUBJECTS_LEG_12
                else:
                    proto_splits = cls._SPLIT_SUBJECTS_DEFAULT
                universe = {s for subs in proto_splits.values() for s in subs}
                held_out = set(split_subjects["val"]) | set(split_subjects["test"])
                split_subjects["train"] = sorted(universe - held_out)

            cls._validate_subject_splits(split_subjects, protocol_mode)
        return activity_ids, split_subjects

    @staticmethod
    def _normalize_subjects(subjects: Iterable[int] | None) -> list[int] | None:
        if subjects is None:
            return None
        out = sorted({int(x) for x in subjects})
        return out

    @classmethod
    def _validate_subject_splits(
        cls, split_subjects: dict[str, list[int]], protocol_mode: str
    ):
        for key in ("train", "val", "test"):
            if len(split_subjects.get(key, [])) == 0:
                raise ValueError(f"{key}_subjects resolved to an empty list.")

        allowed = set(
            cls._SPLIT_SUBJECTS_DEFAULT["train"]
            + cls._SPLIT_SUBJECTS_DEFAULT["val"]
            + cls._SPLIT_SUBJECTS_DEFAULT["test"]
        )
        if protocol_mode in {"ssl", "ssl_wearables"}:
            allowed.discard(109)

        train_set = set(split_subjects["train"])
        val_set = set(split_subjects["val"])
        test_set = set(split_subjects["test"])
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError("Subject splits must be disjoint across train/val/test.")
        all_used = train_set | val_set | test_set
        if not all_used.issubset(allowed):
            bad = sorted(all_used - allowed)
            raise ValueError(
                f"Unknown subjects in split override: {bad}. Allowed: {sorted(allowed)}"
            )

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
