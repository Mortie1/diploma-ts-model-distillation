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
from src.datasets.download import maybe_download_mitbih

logger = logging.getLogger(__name__)


class MITBIHArrhythmiaDataset(BaseDataset):
    """
    MIT-BIH Arrhythmia: binary 10s windows (normal vs arrhythmia).
    """

    DS1 = [
        "101",
        "106",
        "108",
        "109",
        "112",
        "114",
        "115",
        "116",
        "118",
        "119",
        "122",
        "124",
        "201",
        "203",
        "205",
        "207",
        "208",
        "209",
        "215",
        "220",
        "223",
        "230",
    ]
    DS2 = [
        "100",
        "103",
        "105",
        "111",
        "113",
        "117",
        "121",
        "123",
        "200",
        "202",
        "210",
        "212",
        "213",
        "214",
        "219",
        "221",
        "222",
        "228",
        "231",
        "232",
        "233",
        "234",
    ]
    NORMAL_BEAT_SYMBOLS = {"N", "L", "R", "e", "j"}

    def __init__(
        self,
        root: str,
        split: str,
        window_seconds: int = 10,
        target_hz: int = 250,
        stride_seconds: int = 5,
        normalize: bool = True,
        train_val_ratio: float = 0.9,
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")
        if window_seconds <= 0 or target_hz <= 0 or stride_seconds <= 0:
            raise ValueError("window_seconds/target_hz/stride_seconds must be > 0")
        if not (0.0 < train_val_ratio < 1.0):
            raise ValueError("train_val_ratio must be in (0,1)")

        root_path = Path(root)
        if not any(root_path.glob("*.hea")):
            logger.info("MIT-BIH not found at %s, downloading...", root_path)
            maybe_download_mitbih(root_path)
        if not any(root_path.glob("*.hea")):
            nested = list(root_path.rglob("*.hea"))
            if nested:
                src_root = nested[0].parent
                root_path = src_root
        if not any(root_path.glob("*.hea")):
            raise FileNotFoundError(f"MIT-BIH records not found under {root_path}")

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "mitbih_arrhythmia",
            "split": split_key,
            "window_seconds": int(window_seconds),
            "target_hz": int(target_hz),
            "stride_seconds": int(stride_seconds),
            "normalize": bool(normalize),
            "train_val_ratio": float(train_val_ratio),
            "root": str(root_path.resolve()),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "mitbih_arrhythmia" / split_key / signature
        cached = load_cached_index(cache_dir)
        if cached is not None:
            logger.info("MIT-BIH cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array([int(x["label"]) for x in cached], dtype=np.int64)
            super().__init__(index=cached, *args, **kwargs)
            return
        logger.info("MIT-BIH cache miss: split=%s -> %s", split_key, cache_dir)

        ds1 = list(self.DS1)
        n_train = int(round(len(ds1) * train_val_ratio))
        train_records = ds1[:n_train]
        val_records = ds1[n_train:]
        split_records = {
            "train": train_records,
            "val": val_records,
            "test": list(self.DS2),
        }[split_key]

        samples: list[tuple[np.ndarray, int]] = []
        for rec in split_records:
            x, y = self._load_record_windows(
                root_path=root_path,
                record_id=rec,
                window_seconds=window_seconds,
                target_hz=target_hz,
                stride_seconds=stride_seconds,
            )
            for xi, yi in zip(x, y):
                if normalize:
                    mean = xi.mean(axis=1, keepdims=True)
                    std = xi.std(axis=1, keepdims=True) + 1e-6
                    xi = (xi - mean) / std
                samples.append((xi.astype(np.float32), int(yi)))

        if not samples:
            raise RuntimeError("No MIT-BIH windows were generated.")

        self.labels = np.asarray([int(y) for _, y in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(y)) for x, y in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching mitbih/{split_key}",
        )
        super().__init__(index=index, *args, **kwargs)

    @classmethod
    def _load_record_windows(
        cls,
        root_path: Path,
        record_id: str,
        window_seconds: int,
        target_hz: int,
        stride_seconds: int,
    ) -> tuple[list[np.ndarray], list[int]]:
        try:
            import wfdb
        except ImportError as e:
            raise ImportError("MITBIHArrhythmiaDataset requires wfdb") from e

        rec_path = root_path / record_id
        sig, fields = wfdb.rdsamp(str(rec_path))
        ann = wfdb.rdann(str(rec_path), "atr")

        fs_src = int(round(float(fields["fs"])))
        sig1 = sig[:, 0].astype(np.float32)  # MLII-like first channel
        sig1 = cls._resample_1d(sig1, fs_src=fs_src, fs_dst=target_hz)

        ann_samples = np.asarray(ann.sample, dtype=np.float64)
        ann_sym = list(ann.symbol)
        ann_samples = np.round(ann_samples * (float(target_hz) / float(fs_src))).astype(
            np.int64
        )
        ann_samples = np.clip(ann_samples, 0, len(sig1) - 1)

        win = int(window_seconds * target_hz)
        step = int(stride_seconds * target_hz)
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for start in range(0, len(sig1) - win + 1, step):
            end = start + win
            # window label: arrhythmia if any non-normal beat occurs inside.
            mask = (ann_samples >= start) & (ann_samples < end)
            sym = [ann_sym[i] for i in np.where(mask)[0].tolist()]
            if not sym:
                continue
            is_arr = any(s not in cls.NORMAL_BEAT_SYMBOLS for s in sym)
            label = 1 if is_arr else 0
            xw = sig1[start:end][None, :].astype(np.float32)  # [1,T]
            xs.append(xw)
            ys.append(label)
        return xs, ys

    @staticmethod
    def _resample_1d(x: np.ndarray, fs_src: int, fs_dst: int) -> np.ndarray:
        if fs_src == fs_dst:
            return x.astype(np.float32, copy=False)
        n = len(x)
        t_src = np.arange(n, dtype=np.float32) / float(fs_src)
        n_dst = max(1, int(round(n * float(fs_dst) / float(fs_src))))
        t_dst = np.arange(n_dst, dtype=np.float32) / float(fs_dst)
        y = np.interp(t_dst, t_src, x).astype(np.float32)
        return y

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
