from __future__ import annotations

import io
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

UCR_URL_TEMPLATE = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
PAMAP2_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
)
PTBXL_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
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


def ensure_ucr_dataset_layout(dataset_name: str, root: Path) -> Path:
    """
    Normalize UCR on-disk layout to: <root>/<dataset_name>/<dataset_name>_{TRAIN,TEST}.<ext>
    If legacy flat files exist under <root>, move them into the dataset subdirectory.
    """
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for split in ("TRAIN", "TEST"):
        for ext in ("tsv", "txt", "arff"):
            flat = root / f"{dataset_name}_{split}.{ext}"
            nested = dataset_dir / f"{dataset_name}_{split}.{ext}"
            if flat.exists() and not nested.exists():
                logger.info("UCR layout migration: %s -> %s", flat, nested)
                flat.rename(nested)
    return dataset_dir


def maybe_download_ucr(dataset_name: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_ucr_dataset_layout(dataset_name=dataset_name, root=root)
    if any(dataset_dir.glob(f"{dataset_name}_TRAIN.*")) and any(
        dataset_dir.glob(f"{dataset_name}_TEST.*")
    ):
        logger.info("UCR dataset `%s` already present at %s", dataset_name, dataset_dir)
        return
    url = UCR_URL_TEMPLATE.format(name=dataset_name)
    logger.info("Downloading UCR dataset `%s` from %s", dataset_name, url)
    blob = download_bytes(url)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(root)
    ensure_ucr_dataset_layout(dataset_name=dataset_name, root=root)
    logger.info("Extracted UCR dataset `%s` into %s", dataset_name, dataset_dir)


def maybe_download_pamap2(root: Path) -> None:
    protocol_dir = root / "PAMAP2_Dataset" / "Protocol"
    if protocol_dir.exists():
        logger.info("PAMAP2 already present at %s", protocol_dir)
        return

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "PAMAP2_Dataset.zip"
    logger.info("Downloading PAMAP2 from %s", PAMAP2_URL)
    with urllib.request.urlopen(PAMAP2_URL) as resp:  # nosec B310
        total = int(resp.headers.get("Content-Length", "0") or 0)
        with zip_path.open("wb") as f:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="download pamap2",
            ) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    logger.info("Extracting PAMAP2 archive %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    logger.info("PAMAP2 extracted to %s", root)


def maybe_download_ptbxl(root: Path) -> None:
    if (root / "ptbxl_database.csv").exists():
        logger.info("PTB-XL already present at %s", root)
        return

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "ptbxl.zip"
    logger.info("Downloading PTB-XL from %s", PTBXL_URL)
    with urllib.request.urlopen(PTBXL_URL) as resp:  # nosec B310
        total = int(resp.headers.get("Content-Length", "0") or 0)
        with zip_path.open("wb") as f:
            with tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="download ptbxl",
            ) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    logger.info("Extracting PTB-XL archive %s", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    logger.info("PTB-XL extracted to %s", root)


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
