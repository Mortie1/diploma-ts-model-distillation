from __future__ import annotations

import ast
import csv
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
from src.datasets.download import maybe_download_ptbxl

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseDataset):
    """
    PTB-XL ECG classification dataset with predefined fold split.

    Returns:
        inputs: Tensor [C, T] (typically 12 leads)
        targets: int class id
    """

    def __init__(
        self,
        root: str,
        split: str,
        sampling_rate: int = 100,
        label_mode: str = "diagnostic_superclass",
        cache_root: str | None = None,
        *args,
        **kwargs,
    ):
        split_key = split.lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train/val/test")
        if sampling_rate not in {100, 500}:
            raise ValueError("sampling_rate must be 100 or 500")
        if label_mode != "diagnostic_superclass":
            raise ValueError("Only label_mode='diagnostic_superclass' is supported")

        root_path = Path(root)
        self._prepare_ptbxl_root(root_path)
        db_csv = root_path / "ptbxl_database.csv"
        scp_csv = root_path / "scp_statements.csv"
        if not db_csv.exists() or not scp_csv.exists():
            raise FileNotFoundError(
                f"PTB-XL metadata not found under {root_path}. "
                "Expected ptbxl_database.csv and scp_statements.csv."
            )

        cache_base = (
            Path(cache_root)
            if cache_root
            else (root_path / ".cache" / "classification")
        )
        cfg = {
            "dataset": "ptbxl",
            "split": split_key,
            "sampling_rate": int(sampling_rate),
            "label_mode": str(label_mode),
            "db_mtime_ns": db_csv.stat().st_mtime_ns,
            "scp_mtime_ns": scp_csv.stat().st_mtime_ns,
            "root": str(root_path.resolve()),
        }
        signature = cache_signature(cfg)
        cache_dir = cache_base / "ptbxl" / split_key / signature
        cached_index = load_cached_index(cache_dir)
        if cached_index is not None:
            logger.info("PTB-XL cache hit: split=%s path=%s", split_key, cache_dir)
            self.labels = np.array(
                [int(x["label"]) for x in cached_index], dtype=np.int64
            )
            super().__init__(index=cached_index, *args, **kwargs)
            return
        logger.info(
            "PTB-XL cache miss: split=%s -> building cache at %s", split_key, cache_dir
        )

        diag_map = self._load_diagnostic_superclass_map(scp_csv)
        rows = self._load_ptbxl_rows(
            db_csv, split=split_key, sampling_rate=sampling_rate
        )

        samples: list[tuple[np.ndarray, int]] = []
        labels_seen: set[str] = set()
        for row in rows:
            scp_codes = ast.literal_eval(row["scp_codes"])
            label_name = self._pick_superclass_label(scp_codes, diag_map)
            if label_name is None:
                continue
            labels_seen.add(label_name)

        if not labels_seen:
            raise RuntimeError(
                "No PTB-XL samples with diagnostic superclass labels found."
            )
        label_to_idx = {name: i for i, name in enumerate(sorted(labels_seen))}

        for row in rows:
            scp_codes = ast.literal_eval(row["scp_codes"])
            label_name = self._pick_superclass_label(scp_codes, diag_map)
            if label_name is None:
                continue
            signal_path = root_path / row["filename"]
            x = self._load_ecg(signal_path)  # [T, C]
            if x.ndim != 2:
                continue
            x = x.T.astype(np.float32)  # [C, T]
            samples.append((x, int(label_to_idx[label_name])))

        if not samples:
            raise RuntimeError(
                "No PTB-XL samples were materialized. Check download/files/labels."
            )

        self.labels = np.asarray([int(label) for _, label in samples], dtype=np.int64)
        index = materialize_tensor_cache(
            cache_dir=cache_dir,
            samples_with_labels=((x, int(label)) for x, label in samples),
            meta={**cfg, "signature": signature, "n_samples": int(len(samples))},
            total=int(len(samples)),
            progress_desc=f"caching ptbxl/{split_key}",
        )
        super().__init__(index=index, *args, **kwargs)

    @staticmethod
    def _prepare_ptbxl_root(root_path: Path) -> None:
        if (root_path / "ptbxl_database.csv").exists():
            return
        maybe_download_ptbxl(root_path)
        if (root_path / "ptbxl_database.csv").exists():
            return
        # Handle archives that unpack to a nested directory.
        found = list(root_path.rglob("ptbxl_database.csv"))
        if not found:
            return
        nested_root = found[0].parent
        for item in nested_root.iterdir():
            dst = root_path / item.name
            if dst.exists():
                continue
            item.rename(dst)

    @staticmethod
    def _load_diagnostic_superclass_map(path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("diagnostic", "")).strip() not in {
                    "1",
                    "1.0",
                    "True",
                    "true",
                }:
                    continue
                code = str(row.get("scp_code", "")).strip()
                superclass = str(row.get("diagnostic_class", "")).strip()
                if code and superclass:
                    mapping[code] = superclass
        return mapping

    @staticmethod
    def _load_ptbxl_rows(path: Path, split: str, sampling_rate: int) -> list[dict]:
        split_to_folds = {
            "train": set(range(1, 9)),
            "val": {9},
            "test": {10},
        }
        filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        rows: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fold = int(float(row["strat_fold"]))
                except Exception:
                    continue
                if fold not in split_to_folds[split]:
                    continue
                filename = str(row.get(filename_col, "")).strip()
                if not filename:
                    continue
                row = dict(row)
                row["filename"] = filename
                rows.append(row)
        return rows

    @staticmethod
    def _pick_superclass_label(
        scp_codes: dict[str, float],
        code_to_superclass: dict[str, str],
    ) -> str | None:
        candidates: dict[str, float] = {}
        for code, weight in scp_codes.items():
            superclass = code_to_superclass.get(str(code))
            if superclass is None:
                continue
            candidates[superclass] = candidates.get(superclass, 0.0) + float(weight)
        if not candidates:
            return None
        return max(candidates.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _load_ecg(record_path: Path) -> np.ndarray:
        try:
            import wfdb
        except ImportError as e:
            raise ImportError(
                "PTBXLDataset requires `wfdb`. Install it with: pip install wfdb"
            ) from e
        signal, _ = wfdb.rdsamp(str(record_path))
        return signal.astype(np.float32)

    def load_object(self, path):
        return load_tensor_file(path)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        inputs = self.load_object(data_dict["path"])
        targets = int(data_dict["label"])
        instance_data = {"inputs": inputs, "targets": targets}
        instance_data = self.preprocess_data(instance_data)
        return instance_data
