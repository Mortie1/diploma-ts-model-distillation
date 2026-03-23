from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

UCR_URL_TEMPLATE = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}
HF_LTSF_DATASET_REPO = "thuml/Time-Series-Library"
HF_LTSF_CSV_FILES = {
    "etth1": "ETT-small/ETTh1.csv",
    "etth2": "ETT-small/ETTh2.csv",
    "ettm1": "ETT-small/ETTm1.csv",
    "ettm2": "ETT-small/ETTm2.csv",
    "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv",
    "weather": "weather/weather.csv",
    "exchange_rate": "exchange_rate/exchange_rate.csv",
    "national_illness": "illness/national_illness.csv",
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
    dataset_key = dataset_name.lower()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_key in HF_LTSF_CSV_FILES:
        cached_file = hf_hub_download(
            repo_id=HF_LTSF_DATASET_REPO,
            repo_type="dataset",
            filename=HF_LTSF_CSV_FILES[dataset_key],
        )
        shutil.copyfile(cached_file, csv_path)
        return

    if dataset_name not in ETT_URLS:
        raise FileNotFoundError(
            f"Missing dataset file {csv_path} and no auto-download URL for `{dataset_name}`."
        )
    with urllib.request.urlopen(ETT_URLS[dataset_name]) as resp:  # nosec B310
        with csv_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
