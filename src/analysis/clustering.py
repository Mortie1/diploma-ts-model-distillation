from __future__ import annotations

import numpy as np


def cluster_meta_features(features: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError("scikit-learn is required for clustering") from exc

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    return km.fit_predict(features)
