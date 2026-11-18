"""Forecast & Interval Evaluation Metrics.
Point metrics: MAE, sMAPE.
Interval metrics: Empirical Coverage, Normalized Interval Width, Winkler Score.
"""
from typing import Dict
import numpy as np
import pandas as pd


def compute_point_metrics(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    """Computes MAE and sMAPE."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_hat) & (y_true > 0)
    if not np.any(mask):
        return {"mae": np.nan, "smape": np.nan}
    yt = y_true[mask]
    yh = y_hat[mask]
    mae = float(np.mean(np.abs(yt - yh)))
    denom = np.abs(yt) + np.abs(yh)
    smape = float(np.mean(200.0 * np.abs(yt - yh) / np.maximum(1e-6, denom)))
    return {"mae": mae, "smape": smape}


def compute_winkler_score(lo: np.ndarray, hi: np.ndarray, y_true: np.ndarray, level: float) -> float:
    """
    Computes Winkler score for nominal level (1 - alpha):
    W_alpha = (hi - lo) + 2/alpha * (lo - y)*I(y < lo) + 2/alpha * (y - hi)*I(y > hi)
    Lower score indicates sharper, well-calibrated intervals.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(lo) & ~np.isnan(hi)
    if not np.any(mask):
        return np.nan
    l = lo[mask]
    u = hi[mask]
    y = y_true[mask]
    alpha = max(1e-4, 1.0 - level)

    width = u - l
    under = np.maximum(0.0, l - y)
    over = np.maximum(0.0, y - u)
    penalized = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    return float(np.mean(penalized))


def compute_interval_metrics(intervals_df: pd.DataFrame, level: float) -> Dict[str, float]:
    """Computes coverage, mean width, normalized width, and Winkler score."""
    valid = intervals_df.dropna(subset=["lo", "hi", "y_true"]).copy()
    valid = valid[valid["y_true"] > 0]
    n = len(valid)
    if n == 0:
        return {"n": 0, "coverage": np.nan, "mean_width": np.nan, "norm_width": np.nan, "winkler": np.nan}

    cov = float(np.mean((valid["lo"] <= valid["y_true"]) & (valid["y_true"] <= valid["hi"])))
    widths = valid["hi"] - valid["lo"]
    mean_w = float(np.mean(widths))
    norm_w = float(np.mean(widths / np.maximum(1.0, valid["y_true"])))
    winkler = compute_winkler_score(valid["lo"].values, valid["hi"].values, valid["y_true"].values, level)

    return {
        "n": n,
        "coverage": cov,
        "mean_width": mean_w,
        "norm_width": norm_w,
        "winkler": winkler,
    }
