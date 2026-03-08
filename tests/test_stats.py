"""Tests for core.stats."""

import numpy as np

from core.stats import two_sample_ttest


def test_two_sample_ttest():
    np.random.seed(42)
    control = np.random.normal(50, 10, 100)
    treatment = np.random.normal(55, 10, 100)

    result = two_sample_ttest(control, treatment)

    assert 0 <= result.p_value <= 1
    assert result.confidence_interval[0] < result.confidence_interval[1]
    assert isinstance(result.is_significant, bool)
