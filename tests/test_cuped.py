"""Tests for core.cuped."""

import numpy as np

from core.cuped import apply_cuped


def test_cuped_reduces_treatment_variance():
    np.random.seed(42)
    n = 100
    pre_control = np.random.normal(100, 10, n)
    post_control = pre_control + np.random.normal(0, 3, n)
    pre_treatment = np.random.normal(100, 10, n)
    post_treatment = pre_treatment + np.random.normal(0, 3, n)

    result = apply_cuped(pre_control, post_control, pre_treatment, post_treatment)

    var_post = np.var(post_treatment, ddof=1)
    var_adjusted = np.var(result.adjusted_treatment, ddof=1)
    assert var_adjusted < var_post
