"""
ExperimentIQ FastAPI application.

POST endpoints for t-test, sequential test, CUPED, causal forest ATE,
and synthetic control. GET /health for liveness.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Core imports
from core.stats import two_sample_ttest, sequential_test
from core.cuped import apply_cuped
from core.causal import causal_forest_ate
from core.synthetic_control import synthetic_control

app = FastAPI(
    title="ExperimentIQ API",
    description="Statistical experimentation and causal inference endpoints",
    version="0.1.0",
)


# --- Request bodies for each endpoint ---


class TtestRequest(BaseModel):
    control: list[float] = Field(..., description="Control group outcomes")
    treatment: list[float] = Field(..., description="Treatment group outcomes")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")
    equal_var: bool = Field(True, description="Assume equal variances (Welch if False)")


class SequentialRequest(BaseModel):
    control: list[float] = Field(..., description="Control group outcomes")
    treatment: list[float] = Field(..., description="Treatment group outcomes")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")
    total_looks: int = Field(1, ge=1, le=20, description="Total number of looks")
    current_look: int = Field(1, ge=1, le=20, description="Current look index")


class CupedRequest(BaseModel):
    pre_control: list[float] = Field(..., description="Pre-experiment metric, control")
    post_control: list[float] = Field(..., description="Post-experiment metric, control")
    pre_treatment: list[float] = Field(..., description="Pre-experiment metric, treatment")
    post_treatment: list[float] = Field(..., description="Post-experiment metric, treatment")
    theta_from_pooled: bool = Field(False, description="Estimate theta from pooled data")
    run_ttest: bool = Field(True, description="Run t-test on adjusted data and include in response")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Alpha for t-test if run_ttest=True")


class CausalRequest(BaseModel):
    X: list[list[float]] = Field(..., description="Covariates (n_samples x n_features)")
    T: list[float] = Field(..., description="Treatment binary (0/1)")
    Y: list[float] = Field(..., description="Outcome")
    random_state: int | None = Field(None, description="Random state for reproducibility")
    n_estimators: int | None = Field(None, ge=4, description="Number of trees (must be divisible by 4)")


class SyntheticControlRequest(BaseModel):
    data: list[dict[str, Any]] = Field(
        ...,
        description="Long-format rows with time, unit, outcome (keys configurable via config)",
    )
    time_col: str = Field("time", description="Name of time column")
    unit_col: str = Field("unit", description="Name of unit column")
    outcome_col: str = Field("outcome", description="Name of outcome column")
    treated_unit: str = Field("treated", description="Unit identifier for treated")
    treatment_date: str = Field(..., description="First post-treatment date (inclusive)")
    control_units: list[str] | None = Field(None, description="Donor units; default all except treated")


# --- Endpoints ---


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness/readiness check."""
    return {"status": "ok"}


@app.post("/experiment/ttest", response_model=dict[str, Any])
def experiment_ttest(body: TtestRequest) -> dict[str, Any]:
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
    return {
        "p_value": result.p_value,
        "confidence_interval": list(result.confidence_interval),
        "effect_size": result.effect_size,
        "is_significant": result.is_significant,
    }


@app.post("/experiment/sequential", response_model=dict[str, Any])
def experiment_sequential(body: SequentialRequest) -> dict[str, Any]:
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
    return {
        "p_value": result.p_value,
        "confidence_interval": list(result.confidence_interval),
        "effect_size": result.effect_size,
        "is_significant": result.is_significant,
    }


@app.post("/experiment/cuped", response_model=dict[str, Any])
def experiment_cuped(body: CupedRequest) -> dict[str, Any]:
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

    out: dict[str, Any] = {
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
        out["test_result"] = {
            "p_value": ttest_result.p_value,
            "confidence_interval": list(ttest_result.confidence_interval),
            "effect_size": ttest_result.effect_size,
            "is_significant": ttest_result.is_significant,
        }
    return out


@app.post("/experiment/causal", response_model=dict[str, Any])
def experiment_causal(body: CausalRequest) -> dict[str, Any]:
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
        import numpy as np

        X = np.array(body.X)
        T = np.array(body.T)
        Y = np.array(body.Y)
        ate = causal_forest_ate(X, T, Y, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ate": ate}


@app.post("/experiment/synthetic-control", response_model=dict[str, Any])
def experiment_synthetic_control(body: SyntheticControlRequest) -> dict[str, Any]:
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

    return {
        "treatment_effect": [float(x) for x in result["treatment_effect"]],
        "weights": result["weights"],
        "synthetic_outcome": [float(x) for x in result["synthetic_outcome"]],
        "pre_mse": result["pre_mse"],
    }
