import numpy as np

from src.analysis.bootstrap import bootstrap_ci


def test_bootstrap_ci_contains_mean():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    mean, low, high = bootstrap_ci(values, n_resamples=200, random_state=0)
    assert low <= mean <= high
