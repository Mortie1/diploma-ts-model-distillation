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
from src.datasets.download import maybe_download_hhar

logger = logging.getLogger(__name__)


class HHARDataset(BaseDataset):
    """
    HHAR (Heterogeneity Activity Recognition) wrist accelerometer dataset.

    Uses the UCI archive and builds windows from x/y/z accelerometer channels.
    """

    def __init__(
        self,
        root: str,
        split: str,
        window_length: int = 100,
        stride: int = 50,
        upsample_factor: int = 2,
        max_samples: int | None = None,
        sampling_seed: int = 42,
        fold_index: int = 0,
        n_folds: int = 5,
        normalize: bool = True,
        device_filter: str = "watch",
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
        if max_samples is not None and int(max_samples) <= 0:
            raise ValueError("max_samples must be > 0 when provided")
        if n_folds <= 1:
            raise ValueError("n_folds must be > 1")
        if fold_index < 0 or fold_index >= n_folds:
            raise ValueError("fold_index must be in [0, n_folds)")

        root_path = Path(root)
        if not any(root_path.rglob("*.csv")):
            logger.info("HHAR csv files not found under %s, downloading...", root_path)
            maybe_download_hhar(root_path)

        # Filter CSVs by filename to match the device type (watch vs phone).
        # Prefer canonical HHAR files when present to avoid mixing unrelated CSVs
        # from other subdirectories under the same root.
        filter_lower = str(device_filter).lower()
        all_csv_paths = [
            p for p in root_path.rglob("*.csv") if "__macosx" not in str(p).lower()
        ]
        canonical_watch = [
            p for p in all_csv_paths if p.name.lower() == "watch_accelerometer.csv"
        ]
        canonical_phone = [
            p for p in all_csv_paths if p.name.lower() == "phones_accelerometer.csv"
        ]
        if filter_lower == "watch":
            csv_paths = (
                canonical_watch
                if canonical_watch
                else [p for p in all_csv_paths if "watch" in p.name.lower()]
            )
        elif filter_lower == "phone":
            csv_paths = (
                canonical_phone
                if canonical_phone
                else [p for p in all_csv_paths if "phone" in p.name.lower()]
            )
        else:
            csv_paths = all_csv_paths
        if not csv_paths:
            raise FileNotFoundError(
                f"No HHAR csv files found for device_filter={device_filter!r} under {root_path}"
            )

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "hhar",
            "schema_version": 4,
            "split": split_key,
            "window_length": int(window_length),
            "stride": int(stride),
            "upsample_factor": int(upsample_factor),
            "max_samples": (None if max_samples is None else int(max_samples)),
            "sampling_seed": int(sampling_seed),
            "fold_index": int(fold_index),
            "n_folds": int(n_folds),
            "normalize": bool(normalize),
            "device_filter": filter_lower,
            "root": str(root_path.resolve()),
            "n_csv": len(csv_paths),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "hhar" / split_key / signature
        cached = load_cached_index(cache_dir)
        if cached is not None:
            logger.info("HHAR cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array([int(x["label"]) for x in cached], dtype=np.int64)
            super().__init__(index=cached, *args, **kwargs)
            return
        logger.info("HHAR cache miss: split=%s -> %s", split_key, cache_dir)

        file_records: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            need = ["x", "y", "z", "user", "gt"]
            if not all(k in cols for k in need):
                continue
            sub = df[[cols["x"], cols["y"], cols["z"], cols["user"], cols["gt"]]].copy()
            sub.columns = ["x", "y", "z", "user", "gt"]
            sub = sub.dropna(subset=["x", "y", "z", "user", "gt"])
            sub = sub[sub["gt"].astype(str).str.lower() != "null"]
            if sub.empty:
                continue
            user_arr = sub["user"].astype(str).to_numpy()
            # HHAR paper-like: build windows per-file/per-session, no cross-file stitching.
            x_arr = sub[["x", "y", "z"]].to_numpy(dtype=np.float32)  # [T,3]
            y_arr = sub["gt"].astype(str).to_numpy()
            file_records.append((str(p), user_arr, x_arr, y_arr))

        if not file_records:
            raise RuntimeError("No HHAR-compatible csv tables were found.")

        users = sorted(
            {u for _, user_arr, _, _ in file_records for u in np.unique(user_arr)}
        )
        folds = np.array_split(np.array(users, dtype=object), n_folds)
        test_users = set(folds[fold_index].tolist())
        val_users = set(folds[(fold_index + 1) % n_folds].tolist())
        train_users = set(users) - test_users - val_users
        split_users = {"train": train_users, "val": val_users, "test": test_users}[
            split_key
        ]

        split_records = []
        for rec in file_records:
            _, user_arr, _, _ = rec
            if user_arr.size == 0:
                continue
            # Keep only files that belong to selected users for this split.
            # In HHAR each file is typically single-user; if mixed, this remains safe.
            if any(str(u) in split_users for u in np.unique(user_arr)):
                split_records.append(rec)

        if not split_records:
            raise RuntimeError(
                f"HHAR split={split_key} has no files after user split filtering."
            )

        label_names = sorted(
            {
                lbl
                for _, _, _, y_arr in split_records
                for lbl in np.unique(y_arr.astype(str))
            }
        )
        label_to_idx = {name: idx for idx, name in enumerate(label_names)}

        samples: list[tuple[np.ndarray, int]] = []
        for _, _, x, y in split_records:
            if x.shape[0] < window_length:
                continue
            for start in range(0, x.shape[0] - window_length + 1, stride):
                end = start + window_length
                y_win = y[start:end]
                labels, counts = np.unique(y_win, return_counts=True)
                label = str(labels[int(np.argmax(counts))])
                win = x[start:end]  # [window_length, 3]
                if upsample_factor > 1:
                    win = self._upsample_series(
                        win, factor=upsample_factor
                    )  # [window_length*factor, 3]
                win = win.T.astype(np.float32)  # [3, window_length*upsample_factor]
                if normalize:
                    mean = win.mean(axis=1, keepdims=True)
                    std = win.std(axis=1, keepdims=True) + 1e-6
                    win = (win - mean) / std
                samples.append((win, int(label_to_idx[label])))

        if not samples:
            raise RuntimeError("No HHAR windows were generated.")

        if max_samples is not None and len(samples) > int(max_samples):
            samples = self._stratified_sample(
                samples=samples,
                max_samples=int(max_samples),
                seed=int(sampling_seed),
            )

        self.labels = np.asarray([int(y) for _, y in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(y)) for x, y in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching hhar/{split_key}",
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
    def _stratified_sample(
        samples: list[tuple[np.ndarray, int]], max_samples: int, seed: int
    ) -> list[tuple[np.ndarray, int]]:
        rng = np.random.default_rng(seed)
        by_label: dict[int, list[int]] = {}
        for idx, (_, y) in enumerate(samples):
            by_label.setdefault(int(y), []).append(idx)

        labels = sorted(by_label.keys())
        n_classes = len(labels)
        per_class_base = max_samples // n_classes
        remainder = max_samples % n_classes

        selected_indices: list[int] = []
        for i, label in enumerate(labels):
            idxs = np.array(by_label[label], dtype=np.int64)
            rng.shuffle(idxs)
            take = min(len(idxs), per_class_base + (1 if i < remainder else 0))
            selected_indices.extend(idxs[:take].tolist())

        # Fill any leftover quota from the remaining pool (class-balanced first, then global).
        if len(selected_indices) < max_samples:
            selected_set = set(selected_indices)
            remaining = [i for i in range(len(samples)) if i not in selected_set]
            if remaining:
                remaining = np.array(remaining, dtype=np.int64)
                rng.shuffle(remaining)
                need = max_samples - len(selected_indices)
                selected_indices.extend(remaining[:need].tolist())

        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:max_samples]
        return [samples[i] for i in selected_indices]
