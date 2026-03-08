"""
ExperimentIQ Streamlit dashboard.

Sidebar to select experiment type; CSV upload, Plotly control vs treatment
chart, and results card with effect size, p-value, and plain-English interpretation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path when running: streamlit run dashboard/app.py
if Path(__file__).resolve().parent.parent not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ExperimentIQ", page_icon="📊", layout="wide")

# --- Sidebar: experiment type ---
METHODS = [
    "T-Test",
    "Sequential",
    "CUPED",
    "Causal Forest",
    "Synthetic Control",
]
method = st.sidebar.selectbox("Experiment type", METHODS, index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Upload a CSV and choose columns for your experiment.")

# --- Shared: file upload ---
uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv")
if not uploaded:
    st.info("Upload a CSV file to get started.")
    st.stop()

df = pd.read_csv(uploaded)
cols = list(df.columns)

if df.empty or len(cols) < 2:
    st.error("CSV must have at least two columns.")
    st.stop()

# --- Helpers ---
def plot_control_vs_treatment(control_series: pd.Series, treatment_series: pd.Series) -> None:
    """Plot overlapping distributions (histogram + box) for control vs treatment."""
    combined = pd.DataFrame({
        "value": list(control_series.dropna()) + list(treatment_series.dropna()),
        "group": ["Control"] * len(control_series.dropna()) + ["Treatment"] * len(treatment_series.dropna()),
    })
    fig = px.histogram(
        combined, x="value", color="group", barmode="overlay",
        opacity=0.6, nbins=min(40, max(20, len(combined) // 20)),
        title="Control vs Treatment distribution",
        color_discrete_map={"Control": "#1f77b4", "Treatment": "#ff7f0e"},
    )
    fig.update_layout(bargap=0.1, legend_title="", xaxis_title="Value")
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=control_series.dropna(), name="Control", marker_color="#1f77b4"))
    fig2.add_trace(go.Box(y=treatment_series.dropna(), name="Treatment", marker_color="#ff7f0e"))
    fig2.update_layout(title="Control vs Treatment (box plot)", yaxis_title="Value", showlegend=True)
    st.plotly_chart(fig2, width="stretch")


def render_results_card(
    effect_size: float | None,
    p_value: float | None,
    is_significant: bool | None,
    confidence_interval: tuple[float, float] | None = None,
    extra: str = "",
) -> None:
    """Results card and plain-English interpretation."""
    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        if effect_size is not None:
            st.metric("Effect size", f"{effect_size:.4f}")
    with c2:
        if p_value is not None:
            st.metric("P-value", f"{p_value:.4f}")
    with c3:
        if is_significant is not None:
            st.metric("Significant (α=0.05)", "Yes" if is_significant else "No")
    if confidence_interval is not None:
        st.caption(f"95% CI: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
    if extra:
        st.caption(extra)

    st.markdown("#### Interpretation")
    if p_value is not None and is_significant is not None:
        if is_significant:
            st.success(
                "The difference between treatment and control is **statistically significant** (p < 0.05). "
                "The observed effect is unlikely due to chance; you can treat the treatment effect as real."
            )
        else:
            st.warning(
                "We **cannot** conclude a statistically significant difference (p ≥ 0.05). "
                "The observed difference could be due to random variation; consider more data or a larger effect."
            )
    elif effect_size is not None and p_value is None:
        st.info(f"The estimated average treatment effect is **{effect_size:.4f}**. This is a point estimate; no significance test is reported for this method.")
    else:
        st.info("Run the experiment to see the interpretation.")


# --- T-Test ---
def run_ttest():
    from core.stats import two_sample_ttest

    group_col = st.selectbox("Group column", cols, key="ttest_group")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns available for the metric. Please upload data with at least one numeric column.")
        return
    metric_col = st.selectbox("Metric column", numeric_cols, key="ttest_metric")
    alpha = st.slider("Alpha", 0.01, 0.2, 0.05, 0.01, key="ttest_alpha")
    equal_var = st.checkbox("Assume equal variance", True, key="ttest_var")

    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        st.error("Selected metric column must be numeric.")
        return

    unique_groups = df[group_col].unique()
    if len(unique_groups) != 2:
        st.error("Group column must have exactly two unique values.")
        return

    control = df[df[group_col] == unique_groups[0]][metric_col].dropna().tolist()
    treatment = df[df[group_col] == unique_groups[1]][metric_col].dropna().tolist()

    if not control or not treatment:
        st.error("Both groups must have at least one non-missing value in the metric column.")
        return

    st.caption(f"Control: '{unique_groups[0]}' | Treatment: '{unique_groups[1]}'")
    plot_control_vs_treatment(pd.Series(control), pd.Series(treatment))

    if st.button("Run T-Test", key="btn_ttest"):
        result = two_sample_ttest(control, treatment, alpha=alpha, equal_var=equal_var)
        render_results_card(
            effect_size=result.effect_size,
            p_value=result.p_value,
            is_significant=result.is_significant,
            confidence_interval=result.confidence_interval,
        )


# --- Sequential ---
def run_sequential():
    from core.stats import sequential_test

    c_col = st.selectbox("Control column", cols, key="seq_c")
    t_col = st.selectbox("Treatment column", [x for x in cols if x != c_col], key="seq_t")
    alpha = st.slider("Alpha", 0.01, 0.2, 0.05, 0.01, key="seq_alpha")
    total_looks = st.number_input("Total looks", 1, 20, 1, key="seq_total")
    current_look = st.number_input("Current look", 1, 20, 1, key="seq_current")
    if current_look > total_looks:
        st.warning("Current look should be ≤ total looks.")
        return

    control = df[c_col].dropna()
    treatment = df[t_col].dropna()
    plot_control_vs_treatment(control, treatment)

    if st.button("Run Sequential Test", key="btn_seq"):
        result = sequential_test(
            control.tolist(), treatment.tolist(),
            alpha=alpha, total_looks=total_looks, current_look=current_look,
        )
        render_results_card(
            effect_size=result.effect_size,
            p_value=result.p_value,
            is_significant=result.is_significant,
            confidence_interval=result.confidence_interval,
            extra=f"O'Brien-Fleming bounds (look {current_look} of {total_looks}).",
        )


# --- CUPED ---
def run_cuped():
    from core.cuped import apply_cuped
    from core.stats import two_sample_ttest

    st.caption("CSV should have four columns: pre/post for control and treatment (e.g. one row per user with pre_control, post_control, pre_treatment, post_treatment as separate columns; or use column selector for paired columns).")
    pre_c = st.selectbox("Pre-experiment control column", cols, key="cuped_pre_c")
    post_c = st.selectbox("Post-experiment control column", [x for x in cols if x != pre_c], key="cuped_post_c")
    pre_t = st.selectbox("Pre-experiment treatment column", [x for x in cols if x not in (pre_c, post_c)], key="cuped_pre_t")
    post_t = st.selectbox("Post-experiment treatment column", [x for x in cols if x not in (pre_c, post_c, pre_t)], key="cuped_post_t")
    theta_pooled = st.checkbox("Estimate θ from pooled data", False, key="cuped_pooled")
    alpha = st.slider("Alpha (for t-test on adjusted)", 0.01, 0.2, 0.05, 0.01, key="cuped_alpha")

    pre_control = df[pre_c].dropna().tolist()
    post_control = df[post_c].dropna().tolist()
    pre_treatment = df[pre_t].dropna().tolist()
    post_treatment = df[post_t].dropna().tolist()
    if len(pre_control) != len(post_control):
        st.error("Pre- and post-control columns must have the same length (one row per control unit).")
        st.stop()
    if len(pre_treatment) != len(post_treatment):
        st.error("Pre- and post-treatment columns must have the same length (one row per treatment unit).")
        st.stop()
    if not pre_control or not pre_treatment:
        st.error("Need at least one row each for control and treatment.")
        st.stop()

    # Chart: post control vs post treatment (before CUPED)
    plot_control_vs_treatment(pd.Series(post_control), pd.Series(post_treatment))

    if st.button("Run CUPED", key="btn_cuped"):
        cuped = apply_cuped(pre_control, post_control, pre_treatment, post_treatment, theta_from_pooled=theta_pooled)
        result = two_sample_ttest(cuped.adjusted_control, cuped.adjusted_treatment, alpha=alpha)
        render_results_card(
            effect_size=result.effect_size,
            p_value=result.p_value,
            is_significant=result.is_significant,
            confidence_interval=result.confidence_interval,
            extra=f"θ = {cuped.theta:.4f}, pre_mean = {cuped.pre_mean:.4f}. Results are for CUPED-adjusted outcomes.",
        )


# --- Causal Forest ---
def run_causal():
    try:
        from core.causal import causal_forest_ate
    except ImportError:
        st.error("Causal Forest requires the `econml` package. Install it with: pip install econml")
        return

    st.caption("Select covariate columns (X), treatment column (T, 0/1), and outcome column (Y).")
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 3:
        st.error("Need at least 3 numeric columns (covariates, treatment, outcome).")
        return

    x_cols = st.multiselect("Covariate columns (X)", numeric_cols, default=numeric_cols[: max(1, len(numeric_cols) - 2)], key="causal_x")
    t_col = st.selectbox("Treatment column (T, 0/1)", [c for c in numeric_cols if c not in x_cols], key="causal_t")
    y_col = st.selectbox("Outcome column (Y)", [c for c in numeric_cols if c != t_col and c not in x_cols], key="causal_y")

    if not x_cols:
        st.warning("Select at least one covariate.")
        return

    work = df[x_cols + [t_col, y_col]].dropna()
    if len(work) < 20:
        st.error("Need at least 20 rows with no missing values in selected columns.")
        return
    X = work[x_cols].values
    T = work[t_col].values
    Y = work[y_col].values

    # Chart: outcome by treatment group
    plot_df = pd.DataFrame({"outcome": Y.ravel(), "treatment": ["Treatment" if t == 1 else "Control" for t in T.ravel()]})
    fig = px.histogram(plot_df, x="outcome", color="treatment", barmode="overlay", opacity=0.6, nbins=30,
                      title="Outcome distribution by treatment", color_discrete_map={"Control": "#1f77b4", "Treatment": "#ff7f0e"})
    st.plotly_chart(fig, width="stretch")

    if st.button("Run Causal Forest ATE", key="btn_causal"):
        with st.spinner("Fitting causal forest…"):
            ate = causal_forest_ate(X, T, Y, random_state=42, n_estimators=28)
        render_results_card(
            effect_size=ate,
            p_value=None,
            is_significant=None,
            extra=f"Average Treatment Effect (ATE) = {ate:.4f}.",
        )


# --- Synthetic Control ---
def run_synthetic():
    from core.synthetic_control import synthetic_control

    st.caption("Long-format data: one row per (time, unit). Need time, unit, and outcome columns.")
    time_col = st.selectbox("Time column", cols, key="syn_time")
    unit_col = st.selectbox("Unit column", cols, key="syn_unit")
    outcome_col = st.selectbox("Outcome column", cols, key="syn_outcome")
    treated_unit = st.text_input("Treated unit id", "treated", key="syn_treated")
    treatment_date = st.text_input("Treatment date (first post-treatment period)", "2023-07-01", key="syn_date")

    # Build long-format df with correct dtypes
    sdf = df[[time_col, unit_col, outcome_col]].copy()
    sdf = sdf.rename(columns={time_col: "time", unit_col: "unit", outcome_col: "outcome"})
    sdf["time"] = pd.to_datetime(sdf["time"], errors="coerce")
    sdf = sdf.dropna()

    units = sdf["unit"].unique().tolist()
    if treated_unit not in units:
        st.warning(f"Treated unit '{treated_unit}' not found in column. Units: {units[:10]}{'…' if len(units) > 10 else ''}")

    # Chart: outcome over time by unit
    fig = px.line(sdf, x="time", y="outcome", color="unit", title="Outcome over time by unit")
    st.plotly_chart(fig, width="stretch")

    if st.button("Run Synthetic Control", key="btn_syn"):
        try:
            result = synthetic_control(
                sdf, time_col="time", unit_col="unit", outcome_col="outcome",
                treated_unit=treated_unit, treatment_date=treatment_date,
            )
        except Exception as e:
            st.error(str(e))
            return
        te = result["treatment_effect"]
        effect_avg = float(sum(te) / len(te)) if te else None
        render_results_card(
            effect_size=effect_avg,
            p_value=None,
            is_significant=None,
            extra=f"Mean treatment effect (post-period) = {effect_avg:.4f}. Weights: {result['weights']}. Pre-period MSE: {result['pre_mse']:.4f}.",
        )


# --- Route by method ---
st.title("ExperimentIQ Dashboard")
st.markdown(f"**Method:** {method}")

if method == "T-Test":
    run_ttest()
elif method == "Sequential":
    run_sequential()
elif method == "CUPED":
    run_cuped()
elif method == "Causal Forest":
    run_causal()
else:
    run_synthetic()
