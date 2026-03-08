"""
Pydantic v2 request and response schemas for ExperimentIQ API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Base experiment request (control + treatment + alpha) ---


class ExperimentRequest(BaseModel):
    """Base request for experiment endpoints that use control and treatment arrays."""

    control: list[float] = Field(..., description="Control group outcomes")
    treatment: list[float] = Field(..., description="Treatment group outcomes")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Significance level")


# --- Endpoint-specific request models ---


class TtestRequest(ExperimentRequest):
    """Request for POST /experiment/ttest."""

    equal_var: bool = Field(True, description="Assume equal variances (Welch if False)")


class SequentialRequest(ExperimentRequest):
    """Request for POST /experiment/sequential."""

    total_looks: int = Field(1, ge=1, le=20, description="Total number of looks")
    current_look: int = Field(1, ge=1, le=20, description="Current look index")


class CupedRequest(BaseModel):
    """Request for POST /experiment/cuped."""

    pre_control: list[float] = Field(..., description="Pre-experiment metric, control")
    post_control: list[float] = Field(..., description="Post-experiment metric, control")
    pre_treatment: list[float] = Field(..., description="Pre-experiment metric, treatment")
    post_treatment: list[float] = Field(..., description="Post-experiment metric, treatment")
    theta_from_pooled: bool = Field(False, description="Estimate theta from pooled data")
    run_ttest: bool = Field(True, description="Run t-test on adjusted data and include in response")
    alpha: float = Field(0.05, ge=0.001, le=0.5, description="Alpha for t-test if run_ttest=True")


class CausalRequest(BaseModel):
    """Request for POST /experiment/causal."""

    X: list[list[float]] = Field(..., description="Covariates (n_samples x n_features)")
    T: list[float] = Field(..., description="Treatment binary (0/1)")
    Y: list[float] = Field(..., description="Outcome")
    random_state: int | None = Field(None, description="Random state for reproducibility")
    n_estimators: int | None = Field(None, ge=4, description="Number of trees (must be divisible by 4)")


class SyntheticControlRequest(BaseModel):
    """Request for POST /experiment/synthetic-control."""

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


# --- Unified response model ---


class ExperimentResult(BaseModel):
    """Unified response model for experiment endpoints."""

    method: str = Field(..., description="Endpoint/method name (e.g. ttest, sequential, cuped, causal, synthetic-control)")
    effect_size: float | None = Field(None, description="Effect size (e.g. Cohen's d, ATE); None if not applicable")
    p_value: float | None = Field(None, description="P-value from test; None if not applicable")
    confidence_interval: tuple[float, float] | None = Field(
        None,
        description="Confidence interval for effect; None if not applicable",
    )
    is_significant: bool | None = Field(None, description="Whether result is significant at given alpha; None if not applicable")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Method-specific extra fields")

    model_config = {"extra": "forbid"}
