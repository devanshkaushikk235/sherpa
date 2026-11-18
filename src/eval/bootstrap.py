"""Paired Stationary Block Bootstrap for Coverage & Winkler Confidence Intervals.
Uses 7-day blocks and 2,000 resamples over time-indexed evaluation dates.
Reports honest, paired empirical differences (Arm X - Arm A).
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.eval.metrics import compute_winkler_score


def block_bootstrap_paired_diff(df_arm_x: pd.DataFrame, df_arm_base: pd.DataFrame,
                                level: float, block_days: int = 7,
                                n_resamples: int = 2000, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes paired difference distributions for (Coverage_X - Coverage_Base)
    and (Winkler_X - Winkler_Base) using stationary block bootstrap.
    Returns {metric: (mean_diff, ci_lo_95, ci_hi_95)}.
    """
    rng = np.random.default_rng(seed)

    # Merge on (target_date, horizon)
    merged = pd.merge(
        df_arm_x[["target_date", "horizon", "lo", "hi", "y_true"]].rename(
            columns={"lo": "lo_x", "hi": "hi_x"}
        ),
        df_arm_base[["target_date", "horizon", "lo", "hi"]].rename(
            columns={"lo": "lo_b", "hi": "hi_b"}
        ),
        on=["target_date", "horizon"],
    ).dropna()

    if len(merged) < 20:
        return {
            "cov_diff": (np.nan, np.nan, np.nan),
            "winkler_diff": (np.nan, np.nan, np.nan),
        }

    # Group records by target_date
    dates = np.sort(merged["target_date"].unique())
    n_dates = len(dates)
    if n_dates < block_days * 2:
        return {
            "cov_diff": (np.nan, np.nan, np.nan),
            "winkler_diff": (np.nan, np.nan, np.nan),
        }

    date_to_indices = {d: merged.index[merged["target_date"] == d].values for d in dates}

    # Precalculate per-row coverage and winkler penalty
    y = merged["y_true"].values
    cov_x = (merged["lo_x"].values <= y) & (y <= merged["hi_x"].values)
    cov_b = (merged["lo_b"].values <= y) & (y <= merged["hi_b"].values)

    alpha = 1.0 - level
    w_x = (merged["hi_x"].values - merged["lo_x"].values) + \
          (2.0 / alpha) * np.maximum(0.0, merged["lo_x"].values - y) + \
          (2.0 / alpha) * np.maximum(0.0, y - merged["hi_x"].values)

    w_b = (merged["hi_b"].values - merged["lo_b"].values) + \
          (2.0 / alpha) * np.maximum(0.0, merged["lo_b"].values - y) + \
          (2.0 / alpha) * np.maximum(0.0, y - merged["hi_b"].values)

    cov_diff_samples = []
    wink_diff_samples = []

    n_blocks = int(np.ceil(n_dates / block_days))
    for _ in range(n_resamples):
        # Sample block starting indices
        start_indices = rng.integers(0, n_dates - block_days + 1, size=n_blocks)
        sampled_row_indices = []
        for s in start_indices:
            for b in range(block_days):
                d = dates[s + b]
                sampled_row_indices.extend(date_to_indices[d])

        idx = np.array(sampled_row_indices)
        cov_diff = float(np.mean(cov_x[idx]) - np.mean(cov_b[idx]))
        wink_diff = float(np.mean(w_x[idx]) - np.mean(w_b[idx]))

        cov_diff_samples.append(cov_diff)
        wink_diff_samples.append(wink_diff)

    cov_diff_arr = np.array(cov_diff_samples)
    wink_diff_arr = np.array(wink_diff_samples)

    return {
        "cov_diff": (
            float(np.mean(cov_diff_arr)),
            float(np.percentile(cov_diff_arr, 2.5)),
            float(np.percentile(cov_diff_arr, 97.5)),
        ),
        "winkler_diff": (
            float(np.mean(wink_diff_arr)),
            float(np.percentile(wink_diff_arr, 2.5)),
            float(np.percentile(wink_diff_arr, 97.5)),
        ),
    }
