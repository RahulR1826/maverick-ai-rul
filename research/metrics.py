"""
research/metrics.py
-------------------
Evaluation metrics, all computed from actual predictions.

NASA C-MAPSS / PHM08 scoring function (Saxena et al., 2008, "Damage
propagation modeling for aircraft engine run-to-failure simulation"):

    d_i = RUL_pred_i - RUL_true_i
    s_i = exp(-d_i / 13) - 1   if d_i < 0   (early prediction, milder penalty)
          exp( d_i / 10) - 1   if d_i >= 0  (late prediction, harsher penalty)
    S   = sum_i s_i            (summed over test engines; lower is better, 0 is perfect)

The score is a SUM, so it scales with the number of evaluated engines; it is
only comparable between runs that score the same 100 FD001 test engines.
"""

from typing import Dict

import numpy as np


def _check(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Empty arrays")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Non-finite values in metric inputs")
    return y_true, y_pred


def rmse(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def nasa_score(y_true, y_pred) -> float:
    t, p = _check(y_true, y_pred)
    d = p - t
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def error_summary(y_true, y_pred) -> Dict[str, float]:
    t, p = _check(y_true, y_pred)
    d = p - t
    a = np.abs(d)
    return {
        "rmse": rmse(t, p), "mae": mae(t, p), "nasa_score": nasa_score(t, p),
        "bias_mean_error": float(d.mean()),
        "abs_err_median": float(np.median(a)), "abs_err_p90": float(np.percentile(a, 90)),
        "abs_err_max": float(a.max()),
        "frac_late": float(np.mean(d > 0)), "frac_early": float(np.mean(d < 0)),
        "frac_within_10": float(np.mean(a <= 10)), "frac_within_20": float(np.mean(a <= 20)),
        "max_late_error": float(d.max()), "max_early_error": float(d.min()),
        "n": int(t.size),
    }
