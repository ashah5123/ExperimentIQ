"""Tests for core.causal."""

import numpy as np

from core.causal import causal_forest_ate


def test_causal_forest_ate():
    np.random.seed(42)
    X = np.random.normal(0, 1, (200, 3))
    T = np.random.binomial(1, 0.5, 200)
    Y = 2 * T + np.random.normal(0, 1, 200)

    ate = causal_forest_ate(X, T, Y, random_state=42, n_estimators=28)

    assert isinstance(ate, float)
    assert 0.5 <= ate <= 3.5
