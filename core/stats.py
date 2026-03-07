"""
Statistical testing module for ExperimentIQ.

Provides two-sample t-test, Mann-Whitney U, chi-square, and sequential
testing with O'Brien-Fleming alpha spending bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a statistical test."""

    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    is_significant: bool


def _as_arrays(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to 1D numpy arrays."""
    return np.asarray(control).ravel(), np.asarray(treatment).ravel()


def two_sample_ttest(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
) -> TestResult:
    """
    Two-sample t-test for difference in means.

    Uses Welch's t-test if equal_var=False.
    Effect size is Cohen's d. CI is for the difference in means.
    """
    c, t = _as_arrays(control, treatment)
    res = stats.ttest_ind(t, c, equal_var=equal_var, alternative="two-sided")
    p_value = float(res.pvalue)

    n1, n2 = len(c), len(t)
    m1, m2 = np.mean(c), np.mean(t)
    var1, var2 = np.var(c, ddof=1), np.var(t, ddof=1)
    se_diff = np.sqrt(var1 / n1 + var2 / n2)
    if equal_var:
        df = float(n1 + n2 - 2)
    else:
        df = (se_diff**4) / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        df = float(df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = (float(m2 - m1 - t_crit * se_diff), float(m2 - m1 + t_crit * se_diff))

    # Cohen's d (pooled std)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    effect_size = float((m2 - m1) / pooled_std) if pooled_std > 0 else 0.0

    return TestResult(
        p_value=p_value,
        confidence_interval=ci,
        effect_size=effect_size,
        is_significant=p_value < alpha,
    )


def mann_whitney_test(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
) -> TestResult:
    """
    Mann-Whitney U test (non-parametric two-sample).

    Effect size is rank-biserial correlation. CI is bootstrap CI for
    the median difference (treatment - control).
    """
    c, t = _as_arrays(control, treatment)
    res = stats.mannwhitneyu(t, c, alternative="two-sided", method="auto")
    p_value = float(res.pvalue)

    n1, n2 = len(c), len(t)
    n = n1 + n2
    # Rank-biserial correlation: r = 1 - (2*U)/(n1*n2)
    u = res.statistic
    r = 1.0 - (2.0 * u) / (n1 * n2)
    effect_size = float(np.clip(r, -1.0, 1.0))

    # Bootstrap CI for median difference
    rng = np.random.default_rng()
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        bc = rng.choice(c, size=len(c), replace=True)
        bt = rng.choice(t, size=len(t), replace=True)
        diffs[i] = np.median(bt) - np.median(bc)
    ci = (
        float(np.percentile(diffs, 100 * alpha / 2)),
        float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    )

    return TestResult(
        p_value=p_value,
        confidence_interval=ci,
        effect_size=effect_size,
        is_significant=p_value < alpha,
    )


def chi_square_test(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-square test for two independent proportions (2x2 table).

    Expects binary (0/1) arrays: 1 = success, 0 = failure.
    Effect size is phi. CI is for the difference in proportions
    (treatment - control).
    """
    c, t = _as_arrays(control, treatment)
    if np.setdiff1d(np.union1d(c, t), np.array([0.0, 1.0])).size > 0:
        raise ValueError("chi_square_test expects binary (0/1) arrays")

    # 2x2: rows = control, treatment; cols = failure (0), success (1)
    c_success, c_fail = int(np.sum(c)), int(len(c) - np.sum(c))
    t_success, t_fail = int(np.sum(t)), int(len(t) - np.sum(t))
    table = np.array([[c_fail, c_success], [t_fail, t_success]])
    res = stats.chi2_contingency(table, correction=True)
    chi2, p_value = res[0], float(res[1])

    n = table.sum()
    phi = np.sqrt(chi2 / n) if n > 0 else 0.0
    effect_size = float(phi)

    # CI for difference in proportions
    p_c = c_success / len(c) if len(c) else 0.0
    p_t = t_success / len(t) if len(t) else 0.0
    se = np.sqrt(p_c * (1 - p_c) / len(c) + p_t * (1 - p_t) / len(t)) if (len(c) and len(t)) else 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci = (float(diff - z * se), float(diff + z * se))

    return TestResult(
        p_value=p_value,
        confidence_interval=ci,
        effect_size=effect_size,
        is_significant=p_value < alpha,
    )


# O'Brien-Fleming boundary constant C (two-sided alpha=0.05) by number of looks K
# Bound at look k is C * sqrt(K / k). Values from group sequential design tables.
_OBF_C: dict[int, float] = {
    1: 1.96,
    2: 2.24,
    3: 2.29,
    4: 2.31,
    5: 2.33,
    6: 2.34,
    7: 2.35,
    8: 2.36,
    9: 2.36,
    10: 2.37,
}


def _obf_bound(total_looks: int, current_look: int) -> float:
    """O'Brien-Fleming critical value (z-scale) at current look."""
    if current_look < 1 or current_look > total_looks:
        raise ValueError("current_look must be between 1 and total_looks")
    c = _OBF_C.get(total_looks)
    if c is None:
        c = 1.96 + 0.04 * min(total_looks - 1, 10)
    return c * np.sqrt(total_looks / current_look)


def sequential_test(
    control: Sequence[float] | np.ndarray,
    treatment: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    total_looks: int = 1,
    current_look: int = 1,
) -> TestResult:
    """
    Sequential two-sample test with O'Brien-Fleming alpha spending bounds.

    Compares means via z-statistic and rejects if |z| exceeds the
    O'Brien-Fleming bound for (current_look, total_looks).
    Effect size is Cohen's d; CI is for the difference in means.
    """
    c, t = _as_arrays(control, treatment)
    n1, n2 = len(c), len(t)
    m1, m2 = np.mean(c), np.mean(t)
    var1, var2 = np.var(c, ddof=1), np.var(t, ddof=1)
    se_diff = np.sqrt(var1 / n1 + var2 / n2)
    if se_diff == 0:
        z_stat = 0.0
    else:
        z_stat = (m2 - m1) / se_diff

    bound = _obf_bound(total_looks, current_look)
    is_significant = abs(z_stat) >= bound
    # Nominal two-sided p-value from z
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # CI for difference in means (z-based, same as large-sample t)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (
        float(m2 - m1 - z_crit * se_diff),
        float(m2 - m1 + z_crit * se_diff),
    )

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    effect_size = float((m2 - m1) / pooled_std) if pooled_std > 0 else 0.0

    return TestResult(
        p_value=p_value,
        confidence_interval=ci,
        effect_size=effect_size,
        is_significant=is_significant,
    )
