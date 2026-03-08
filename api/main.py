"""
ExperimentIQ FastAPI application.

POST endpoints for t-test, sequential test, CUPED, causal forest ATE,
and synthetic control. GET /health for liveness.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import (
    CausalRequest,
    CupedRequest,
    ExperimentResult,
    SequentialRequest,
    SyntheticControlRequest,
    TtestRequest,
)
from core.causal import causal_forest_ate
from core.cuped import apply_cuped
from core.stats import sequential_test, two_sample_ttest
from core.synthetic_control import synthetic_control

app = FastAPI(
    title="ExperimentIQ API",
    description="Statistical experimentation and causal inference endpoints",
    version="0.1.0",
)


# --- Endpoints ---


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness check."""
    return {"status": "ok"}


@app.post("/experiment/ttest", response_model=ExperimentResult)
def experiment_ttest(body: TtestRequest) -> ExperimentResult:
    """Two-sample t-test on control vs treatment."""
    try:
        result = two_sample_ttest(
            body.control,
            body.treatment,
            alpha=body.alpha,
            equal_var=body.equal_var,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExperimentResult(
        method="ttest",
        effect_size=result.effect_size,
        p_value=result.p_value,
        confidence_interval=result.confidence_interval,
        is_significant=result.is_significant,
        metadata={"equal_var": body.equal_var, "alpha": body.alpha},
    )


@app.post("/experiment/sequential", response_model=ExperimentResult)
def experiment_sequential(body: SequentialRequest) -> ExperimentResult:
    """Sequential test with O'Brien-Fleming bounds."""
    if body.current_look > body.total_looks:
        raise HTTPException(
            status_code=400,
            detail="current_look must be <= total_looks",
        )
    try:
        result = sequential_test(
            body.control,
            body.treatment,
            alpha=body.alpha,
            total_looks=body.total_looks,
            current_look=body.current_look,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExperimentResult(
        method="sequential",
        effect_size=result.effect_size,
        p_value=result.p_value,
        confidence_interval=result.confidence_interval,
        is_significant=result.is_significant,
        metadata={
            "alpha": body.alpha,
            "total_looks": body.total_looks,
            "current_look": body.current_look,
        },
    )


@app.post("/experiment/cuped", response_model=ExperimentResult)
def experiment_cuped(body: CupedRequest) -> ExperimentResult:
    """CUPED adjustment; optionally run t-test on adjusted data."""
    try:
        cuped_result = apply_cuped(
            body.pre_control,
            body.post_control,
            body.pre_treatment,
            body.post_treatment,
            theta_from_pooled=body.theta_from_pooled,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    effect_size: float | None = None
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    is_significant: bool | None = None
    metadata: dict[str, Any] = {
        "theta": cuped_result.theta,
        "pre_mean": cuped_result.pre_mean,
        "adjusted_control": cuped_result.adjusted_control.tolist(),
        "adjusted_treatment": cuped_result.adjusted_treatment.tolist(),
    }

    if body.run_ttest:
        ttest_result = two_sample_ttest(
            cuped_result.adjusted_control,
            cuped_result.adjusted_treatment,
            alpha=body.alpha,
        )
        effect_size = ttest_result.effect_size
        p_value = ttest_result.p_value
        confidence_interval = ttest_result.confidence_interval
        is_significant = ttest_result.is_significant
        metadata["alpha"] = body.alpha

    return ExperimentResult(
        method="cuped",
        effect_size=effect_size,
        p_value=p_value,
        confidence_interval=confidence_interval,
        is_significant=is_significant,
        metadata=metadata,
    )


@app.post("/experiment/causal", response_model=ExperimentResult)
def experiment_causal(body: CausalRequest) -> ExperimentResult:
    """Causal forest ATE estimation."""
    if not (len(body.X) == len(body.T) == len(body.Y)):
        raise HTTPException(
            status_code=400,
            detail="X, T, and Y must have the same length",
        )
    kwargs: dict[str, Any] = {}
    if body.random_state is not None:
        kwargs["random_state"] = body.random_state
    if body.n_estimators is not None:
        kwargs["n_estimators"] = body.n_estimators
    try:
        X = np.array(body.X)
        T = np.array(body.T)
        Y = np.array(body.Y)
        ate = causal_forest_ate(X, T, Y, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return ExperimentResult(
        method="causal",
        effect_size=ate,
        p_value=None,
        confidence_interval=None,
        is_significant=None,
        metadata={"ate": ate, **kwargs},
    )


@app.post("/experiment/synthetic-control", response_model=ExperimentResult)
def experiment_synthetic_control(body: SyntheticControlRequest) -> ExperimentResult:
    """Synthetic control: weights and treatment effect in post-period."""
    try:
        import pandas as pd

        df = pd.DataFrame(body.data)
        if body.time_col not in df.columns or body.unit_col not in df.columns or body.outcome_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Data must contain columns: {body.time_col}, {body.unit_col}, {body.outcome_col}",
            )
        result = synthetic_control(
            df,
            time_col=body.time_col,
            unit_col=body.unit_col,
            outcome_col=body.outcome_col,
            treated_unit=body.treated_unit,
            treatment_date=body.treatment_date,
            control_units=body.control_units,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    treatment_effect = [float(x) for x in result["treatment_effect"]]
    effect_size = float(np.mean(treatment_effect)) if treatment_effect else None
    return ExperimentResult(
        method="synthetic-control",
        effect_size=effect_size,
        p_value=None,
        confidence_interval=None,
        is_significant=None,
        metadata={
            "treatment_effect": treatment_effect,
            "weights": result["weights"],
            "synthetic_outcome": [float(x) for x in result["synthetic_outcome"]],
            "pre_mse": result["pre_mse"],
        },
    )
