"""Patch-Level Aggregated Metrics.
Aggregates evaluation intervals into discrete patch event clusters within the 0-7d window.
Each row represents one independent patch shock event with its empirical coverage, mean width,
Winkler score, and capacity costs.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from src.eval.metrics import compute_winkler_score


def aggregate_patch_level(intervals_df: pd.DataFrame,
                          ratios: List[float] = [2.0, 5.0, 10.0]) -> pd.DataFrame:
    """
    Computes patch-level aggregated metrics for 0-7d post-patch intervals on EVAL split.
    Returns DataFrame with columns:
    [game_id, patch_id, patch_type, model, arm, level, n_points, coverage, mean_width, winkler, cost_2x, cost_5x, cost_10x]
    """
    # Filter strictly to EVAL split, 0-7d post-patch window, and valid patch_id
    valid = intervals_df[
        (intervals_df["split"] == "EVAL") &
        (intervals_df["in_patch_window"] == True) &
        (intervals_df["days_since_patch"] >= 0) &
        (intervals_df["days_since_patch"] <= 7) &
        (intervals_df["patch_id"] != "none") &
        intervals_df["y_true"].notna()
    ].copy()

    if len(valid) == 0:
        return pd.DataFrame(columns=[
            "game_id", "patch_id", "patch_type", "model", "arm", "level",
            "n_points", "coverage", "mean_width", "winkler",
            "cost_2x", "cost_5x", "cost_10x",
        ])

    records = []
    group_cols = ["game_id", "patch_id", "patch_type", "model", "arm", "level"]
    for (gid, pid, ptype, model, arm, lvl), g in valid.groupby(group_cols):
        n = len(g)
        if n == 0:
            continue
        y = g["y_true"].values
        lo = g["lo"].values
        hi = g["hi"].values

        cov = float(np.mean((lo <= y) & (y <= hi)))
        width = float(np.mean(hi - lo))
        winkler = compute_winkler_score(lo, hi, y, level=float(lvl))

        # Asymmetric capacity costs per patch
        mean_y = float(max(1.0, np.mean(y)))
        under = np.maximum(0.0, y - hi)
        over = np.maximum(0.0, hi - y)

        cost_dict = {}
        for r in ratios:
            cost_dict[f"cost_{int(r)}x"] = float(np.mean(r * under + 1.0 * over) / mean_y)

        rec = {
            "game_id": gid,
            "patch_id": pid,
            "patch_type": ptype,
            "model": model,
            "arm": arm,
            "level": float(lvl),
            "n_points": n,
            "coverage": cov,
            "mean_width": width,
            "norm_width": float(np.mean((hi - lo) / np.maximum(1.0, y))),
            "winkler": winkler,
        }
        rec.update(cost_dict)
        records.append(rec)

    return pd.DataFrame(records)
