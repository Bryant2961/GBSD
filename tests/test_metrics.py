import math

from gbsd.evaluation.metrics import (
    average_interval_width,
    coverage95,
    error_std_corr,
    gaussian_nll,
    relative_l2,
)


def test_relative_l2_perfect_prediction():
    assert relative_l2([1.0, 2.0], [1.0, 2.0]) == 0.0


def test_coverage_and_interval_width():
    pred = [0.0, 0.0]
    target = [0.1, 3.0]
    std = [1.0, 1.0]
    assert coverage95(pred, target, std) == 0.5
    assert math.isclose(average_interval_width(std), 3.92)


def test_nll_is_finite_with_zero_std_floor():
    value = gaussian_nll([0.0], [0.0], [0.0])
    assert math.isfinite(value)


def test_error_std_corr_defined_for_varying_inputs():
    corr = error_std_corr([0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.1, 0.2, 0.3])
    assert corr > 0.99

