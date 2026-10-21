import math

import numpy as np
import pytest

from research import metrics as M


def test_perfect_prediction():
    y = np.array([10.0, 50.0, 125.0])
    assert M.rmse(y, y) == 0.0
    assert M.mae(y, y) == 0.0
    assert M.nasa_score(y, y) == 0.0


def test_rmse_mae_hand_computed():
    t = np.array([0.0, 0.0, 0.0, 0.0])
    p = np.array([1.0, -1.0, 3.0, -3.0])
    assert M.rmse(t, p) == pytest.approx(math.sqrt(5.0))
    assert M.mae(t, p) == pytest.approx(2.0)


def test_nasa_asymmetry_and_constants():
    # d = pred - true. Late (d>0) uses /10, early (d<0) uses /13 (Saxena et al. 2008).
    late = M.nasa_score([50.0], [60.0])   # d = +10
    early = M.nasa_score([50.0], [40.0])  # d = -10
    assert late == pytest.approx(math.e - 1.0)
    assert early == pytest.approx(math.exp(10 / 13) - 1.0)
    assert late > early


def test_nasa_is_sum_over_engines():
    t = np.array([10.0, 20.0])
    p = np.array([23.0, 7.0])          # d = +13, -13
    expected = (math.exp(1.3) - 1) + (math.e - 1)
    assert M.nasa_score(t, p) == pytest.approx(expected)


def test_nasa_zero_boundary_belongs_to_late_branch():
    # d == 0 gives exp(0)-1 = 0 in either branch; tiny positive d uses /10.
    assert M.nasa_score([5.0], [5.0 + 1e-9]) == pytest.approx(1e-10, rel=1e-3)


def test_metric_input_validation():
    with pytest.raises(ValueError):
        M.rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        M.mae([], [])
    with pytest.raises(ValueError):
        M.nasa_score([1.0], [float("nan")])


def test_large_late_error_is_finite_and_huge():
    s = M.nasa_score([0.0], [100.0])
    assert np.isfinite(s) and s == pytest.approx(math.exp(10) - 1)


def test_error_summary_fields():
    s = M.error_summary([10.0, 20.0, 30.0], [12.0, 15.0, 30.0])
    assert s["n"] == 3
    assert s["frac_late"] == pytest.approx(1 / 3)
    assert s["frac_early"] == pytest.approx(1 / 3)
    assert s["max_late_error"] == pytest.approx(2.0)
    assert s["max_early_error"] == pytest.approx(-5.0)
    assert s["bias_mean_error"] == pytest.approx(-1.0)


def test_matches_legacy_implementation():
    pytest.importorskip("torch")
    from training.utils import compute_mae, compute_rmse, nasa_score
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 125, 100)
    p = t + rng.normal(0, 15, 100)
    assert M.rmse(t, p) == pytest.approx(compute_rmse(t, p))
    assert M.mae(t, p) == pytest.approx(compute_mae(t, p))
    assert M.nasa_score(t, p) == pytest.approx(nasa_score(t, p))
