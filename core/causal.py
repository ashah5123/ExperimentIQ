"""
Causal inference module for ExperimentIQ.

Uses econml for causal forest and ATE estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from econml.dml import CausalForestDML


def causal_forest_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    discrete_treatment: bool = True,
    random_state: int | None = None,
    **kwargs: Any,
) -> float:
    """
    Estimate the average treatment effect (ATE) using a causal forest (DML).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Covariates.
    T : array-like of shape (n_samples,)
        Treatment (binary 0/1).
    Y : array-like of shape (n_samples,)
        Outcome.
    discrete_treatment : bool, default True
        Whether treatment is binary/categorical.
    random_state : int or None, default None
        Random state for reproducibility.
    **kwargs
        Passed to CausalForestDML (e.g. n_estimators, max_depth).
        Note: n_estimators must be divisible by subforest_size (default 4).

    Returns
    -------
    float
        ATE estimate (scalar).
    """
    X = np.asarray(X)
    T = np.asarray(T).ravel()
    Y = np.asarray(Y).ravel()

    est = CausalForestDML(
        discrete_treatment=discrete_treatment,
        drate=True,
        random_state=random_state,
        **kwargs,
    )
    est.fit(Y, T, X=X)

    ate = est.ate_
    if np.ndim(ate) > 0:
        ate = float(ate.ravel()[0])
    else:
        ate = float(ate)
    return ate
