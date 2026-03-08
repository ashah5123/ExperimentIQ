"""
Synthetic control method for ExperimentIQ.

Constructs a synthetic control from donor units to estimate
counterfactual and treatment effects for a single treated unit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def synthetic_control(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    unit_col: str = "unit",
    outcome_col: str = "outcome",
    treated_unit: str = "treated",
    treatment_date: pd.Timestamp | str,
    control_units: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run synthetic control: fit weights on control units in the pre-period,
    then compute treatment effect as (treated - synthetic) in the post-period.

    Parameters
    ----------
    df : DataFrame
        Long-format data with time, unit, and outcome columns.
    time_col : str, default "time"
        Name of the time/dates column.
    unit_col : str, default "unit"
        Name of the unit identifier column.
    outcome_col : str, default "outcome"
        Name of the outcome column.
    treated_unit : str, default "treated"
        Value in unit_col that identifies the treated unit.
    treatment_date : str or Timestamp
        First period considered post-treatment (inclusive).
    control_units : list of str or None, default None
        Units to use as donors. If None, all units except treated_unit are used.

    Returns
    -------
    dict
        - treatment_effect : array of treatment effect for each post-treatment period
        - weights : dict of unit -> weight for each donor
        - synthetic_outcome : array of synthetic control outcome in post period
        - pre_mse : pre-treatment mean squared error (fit quality)
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    treatment_date = pd.Timestamp(treatment_date)

    units = df[unit_col].unique().tolist()
    if treated_unit not in units:
        raise ValueError(f"treated_unit '{treated_unit}' not found in {unit_col}")

    if control_units is None:
        control_units = [u for u in units if u != treated_unit]
    else:
        for u in control_units:
            if u not in units:
                raise ValueError(f"control unit '{u}' not found in {unit_col}")

    if not control_units:
        raise ValueError("At least one control unit is required")

    # Pre vs post split
    pre = df[df[time_col] < treatment_date].sort_values(time_col)
    post = df[df[time_col] >= treatment_date].sort_values(time_col)

    # Pre-treatment outcome vectors
    pre_treated = pre.loc[pre[unit_col] == treated_unit, outcome_col].values
    pre_control_matrix = np.column_stack([
        pre.loc[pre[unit_col] == u, outcome_col].values
        for u in control_units
    ])

    n_pre = len(pre_treated)
    n_control = len(control_units)
    if pre_control_matrix.shape[0] != n_pre:
        raise ValueError(
            "Pre-period rows per unit must align. Ensure each unit has one outcome per time in the pre-period."
        )

    # Minimize ||pre_treated - pre_control_matrix @ w||^2 s.t. w >= 0, sum(w) == 1
    def loss(w: np.ndarray) -> float:
        pred = pre_control_matrix @ w
        return float(np.sum((pre_treated - pred) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_control
    w0 = np.ones(n_control) / n_control
    sol = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    weights_vec = np.maximum(sol.x, 0)
    weights_vec = weights_vec / weights_vec.sum()

    pre_mse = float(np.mean((pre_treated - pre_control_matrix @ weights_vec) ** 2))

    # Post-period: synthetic and treatment effect
    post_treated = post.loc[post[unit_col] == treated_unit, outcome_col].values
    post_control_matrix = np.column_stack([
        post.loc[post[unit_col] == u, outcome_col].values
        for u in control_units
    ])
    n_post = len(post_treated)
    if post_control_matrix.shape[0] != n_post:
        raise ValueError(
            "Post-period rows per unit must align. Ensure each unit has one outcome per time in the post-period."
        )
    synthetic_outcome = post_control_matrix @ weights_vec
    treatment_effect = post_treated - synthetic_outcome

    weights_dict = dict(zip(control_units, weights_vec.tolist()))

    return {
        "treatment_effect": treatment_effect,
        "weights": weights_dict,
        "synthetic_outcome": np.asarray(synthetic_outcome),
        "pre_mse": pre_mse,
    }
