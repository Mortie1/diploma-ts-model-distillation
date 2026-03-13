from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.bootstrap import bootstrap_ci


def load_outputs(path: Path):
    files = sorted(path.glob("*.pth"))
    return [torch.load(fp, map_location="cpu") for fp in files]


def eval_classification(outputs):
    correct = []
    for out in outputs:
        pred = int(out["pred_label"])
        label = int(out["label"])
        correct.append(1.0 if pred == label else 0.0)
    values = np.array(correct, dtype=np.float64)
    mean, low, high = bootstrap_ci(values, n_resamples=1000)
    return {"accuracy": mean, "accuracy_ci_low": low, "accuracy_ci_high": high}


def eval_forecasting(outputs):
    maes = []
    for out in outputs:
        forecast = out["forecast"].float()
        target = out["target"].float()
        maes.append((forecast - target).abs().mean().item())
    values = np.array(maes, dtype=np.float64)
    mean, low, high = bootstrap_ci(values, n_resamples=1000)
    return {"mae": mean, "mae_ci_low": low, "mae_ci_high": high}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--task", choices=["classification", "forecasting"], required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    parts = [p for p in pred_dir.iterdir() if p.is_dir()]

    for part_dir in parts:
        outputs = load_outputs(part_dir)
        if not outputs:
            continue
        if args.task == "classification":
            metrics = eval_classification(outputs)
        else:
            metrics = eval_forecasting(outputs)
        print(part_dir.name, metrics)


if __name__ == "__main__":
    main()
