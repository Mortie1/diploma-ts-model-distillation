from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

UCR_URL_TEMPLATE = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}


def download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # nosec B310
        return resp.read()


def maybe_download_ucr(dataset_name: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    url = UCR_URL_TEMPLATE.format(name=dataset_name)
    blob = download_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(root)


def maybe_download_ett(csv_path: Path) -> None:
    dataset_name = csv_path.stem
    if dataset_name not in ETT_URLS:
        raise FileNotFoundError(
            f"Missing dataset file {csv_path} and no auto-download URL for `{dataset_name}`."
        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ETT_URLS[dataset_name]) as resp:  # nosec B310
        with csv_path.open("wb") as f:
            shutil.copyfileobj(resp, f)

