from __future__ import annotations

import numpy as np


def spectral_entropy(x: np.ndarray) -> float:
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    power = power / (power.sum() + 1e-12)
    return float(-(power * np.log(power + 1e-12)).sum())


def stationarity_proxy(x: np.ndarray) -> float:
    # Ratio of variance in first difference to raw variance.
    var_raw = np.var(x) + 1e-12
    var_diff = np.var(np.diff(x))
    return float(var_diff / var_raw)


def series_meta_features(x: np.ndarray) -> dict[str, float]:
    return {
        "length": float(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "spectral_entropy": spectral_entropy(x),
        "stationarity_proxy": stationarity_proxy(x),
    }
