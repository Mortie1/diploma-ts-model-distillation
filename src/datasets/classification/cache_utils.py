from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Iterable

import torch


def cache_signature(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / "index.json"


def load_cached_index(cache_dir: Path) -> list[dict] | None:
    idx_path = _index_path(cache_dir)
    if not idx_path.exists():
        return None
    try:
        records = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(records, list):
        return None
    for rec in records:
        if not isinstance(rec, dict):
            return None
        path = rec.get("path")
        if not path or not Path(path).exists():
            return None
        if "label" not in rec:
            return None
    return records


def materialize_tensor_cache(
    cache_dir: Path,
    samples_with_labels: Iterable[tuple],
    meta: dict | None = None,
) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for idx, (x, label) in enumerate(samples_with_labels):
        tensor = torch.as_tensor(x, dtype=torch.float32)
        out_path = cache_dir / f"{idx:08d}.pt"
        torch.save(tensor, out_path)
        records.append({"path": str(out_path), "label": int(label)})

    if meta is not None:
        (cache_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    idx_path = _index_path(cache_dir)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=cache_dir, suffix=".index.tmp", encoding="utf-8"
    ) as tmp:
        json.dump(records, tmp, ensure_ascii=True)
        tmp.flush()
        tmp_path = Path(tmp.name)
    tmp_path.replace(idx_path)
    return records


def load_tensor_file(path: str | Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")
