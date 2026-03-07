"""
CUPED (Controlled-experiment Using Pre-Experiment Data) for ExperimentIQ.

Variance reduction by regressing out the pre-experiment covariate.
Adjusted arrays can be passed to the stats module for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CupedResult:
    """Result of CUPED adjustment."""

    adjusted_control: np.ndarray
    adjusted_treatment: np.ndarray
    theta: float
    pre_mean: float


def apply_cuped(
    pre_control: Sequence[float] | np.ndarray,
    post_control: Sequence[float] | np.ndarray,
    pre_treatment: Sequence[float] | np.ndarray,
    post_treatment: Sequence[float] | np.ndarray,
    theta_from_pooled: bool = False,
) -> CupedResult:
    """
    Apply CUPED to reduce variance using pre-experiment data as a covariate.

    The adjustment is: Y_adj = Y - θ * (X - X_mean), where θ = Cov(Y, X) / Var(X).
    By default θ is estimated from the control group only to avoid bias; set
    theta_from_pooled=True to estimate from pooled data.

    Parameters
    ----------
    pre_control : Pre-experiment metric for control group.
    post_control : Post-experiment metric for control group.
    pre_treatment : Pre-experiment metric for treatment group.
    post_treatment : Post-experiment metric for treatment group.
    theta_from_pooled : If True, estimate θ from pooled control+treatment;
        if False (default), estimate from control only.

    Returns
    -------
    CupedResult
        adjusted_control, adjusted_treatment : arrays ready for core.stats
        theta : coefficient used for adjustment
        pre_mean : mean of pre-experiment used for centering (pooled)
    """
    pre_c = np.asarray(pre_control, dtype=float).ravel()
    post_c = np.asarray(post_control, dtype=float).ravel()
    pre_t = np.asarray(pre_treatment, dtype=float).ravel()
    post_t = np.asarray(post_treatment, dtype=float).ravel()

    if len(pre_c) != len(post_c) or len(pre_t) != len(post_t):
        raise ValueError("Pre- and post-experiment arrays must have the same length per group")

    # Centering: use pooled pre-experiment mean so both groups are comparable
    pre_all = np.concatenate([pre_c, pre_t])
    pre_mean = float(np.mean(pre_all))

    if theta_from_pooled:
        y_all = np.concatenate([post_c, post_t])
        x_all = np.concatenate([pre_c, pre_t])
        cov_xy = np.cov(x_all, y_all)[0, 1]
        var_x = np.var(x_all, ddof=1)
    else:
        cov_xy = np.cov(pre_c, post_c)[0, 1]
        var_x = np.var(pre_c, ddof=1)

    if var_x <= 0:
        theta = 0.0
    else:
        theta = float(cov_xy / var_x)

    pre_c_centered = pre_c - pre_mean
    pre_t_centered = pre_t - pre_mean
    adjusted_control = post_c - theta * pre_c_centered
    adjusted_treatment = post_t - theta * pre_t_centered

    return CupedResult(
        adjusted_control=adjusted_control,
        adjusted_treatment=adjusted_treatment,
        theta=theta,
        pre_mean=pre_mean,
    )
