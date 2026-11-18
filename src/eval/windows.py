"""Window Stratification & Aggregated Metrics.
Stratifies evaluation rows into:
- 0-7 days post-patch
- 8-14 days post-patch
- 15-30 days post-patch
- baseline (> 30 days or no patch)
And additionally by major vs minor patch types.
"""
from typing import List
import numpy as np
import pandas as pd
from src.eval.metrics import compute_interval_metrics, compute_point_metrics


def assign_stratum(days_since_patch: float) -> str:
    if np.isnan(days_since_patch) or days_since_patch > 30:
        return "baseline (>30d)"
    if 0 <= days_since_patch <= 7:
        return "post_patch (0-7d)"
    if 8 <= days_since_patch <= 14:
        return "post_patch (8-14d)"
    if 15 <= days_since_patch <= 30:
        return "post_patch (15-30d)"
    return "baseline (>30d)"


def aggregate_strata_metrics(intervals_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes metrics across (game_id, model, arm, level, window_stratum).
    """
    df = intervals_df.copy()
    df["window"] = df["days_since_patch"].apply(assign_stratum)

    # Compute point metrics lookup per (game_id, model, window)
    point_lookup = {}
    f_df = forecasts_df.copy()
    if "days_since_patch" in df.columns:
        # map window to forecasts via composite key (game_id, target_date)
        key_win_map = dict(zip(zip(df["game_id"], df["target_date"]), df["window"]))
        f_keys = list(zip(f_df["game_id"], f_df["target_date"]))
        f_df["window"] = [key_win_map.get(k, "baseline (>30d)") for k in f_keys]
        for (gid, m, w), g in f_df.groupby(["game_id", "model", "window"]):
            pm = compute_point_metrics(g["y_true"].values, g["y_hat"].values)
            point_lookup[(gid, m, w)] = pm

    records = []
    group_cols = ["game_id", "model", "arm", "level", "window"]
    for (gid, m, arm, lvl, win), g in df.groupby(group_cols):
        im = compute_interval_metrics(g, level=lvl)
        pm = point_lookup.get((gid, m, win), {"mae": np.nan, "smape": np.nan})
        records.append({
            "game_id":    gid,
            "model":      m,
            "arm":        arm,
            "level":      float(lvl),
            "window":     win,
            "n":          int(im["n"]),
            "coverage":   float(im["coverage"]),
            "mean_width": float(im["mean_width"]),
            "norm_width": float(im["norm_width"]),
            "winkler":    float(im["winkler"]),
            "mae":        float(pm["mae"]),
            "smape":      float(pm["smape"]),
        })

    return pd.DataFrame(records)


def point_forecast_table(forecasts_df: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a standalone point-forecast quality table stratified by post-patch window.
    Columns: game_id, model, window, n, mae, smape, mape.

    This allows readers to distinguish:
      (a) Calibration failure  — point forecast is accurate but intervals are too narrow.
      (b) Model-bias failure   — point forecast degrades after patches, pulling intervals off-centre.

    Must be called on the EVAL split only.
    """
    # Use the interval frame to get the window label per (game_id, target_date)
    win_map = (
        intervals_df[["game_id", "target_date", "days_since_patch"]]
        .drop_duplicates(subset=["game_id", "target_date"])
        .assign(window=lambda d: d["days_since_patch"].apply(assign_stratum))
    )

    f = forecasts_df.merge(
        win_map[["game_id", "target_date", "window"]],
        on=["game_id", "target_date"],
        how="left",
    )
    f["window"] = f["window"].fillna("baseline (>30d)")

    records = []
    for (gid, model, window), g in f.groupby(["game_id", "model", "window"]):
        valid = g.dropna(subset=["y_true", "y_hat"])
        valid = valid[valid["y_true"] > 0]
        if valid.empty:
            continue
        yt = valid["y_true"].values
        yh = valid["y_hat"].values
        mae   = float(np.mean(np.abs(yt - yh)))
        mape  = float(np.mean(np.abs((yt - yh) / yt) * 100))
        denom = np.abs(yt) + np.abs(yh)
        smape = float(np.mean(200.0 * np.abs(yt - yh) / np.maximum(1e-6, denom)))
        records.append({
            "game_id": gid,
            "model":   model,
            "window":  window,
            "n":       len(valid),
            "mae":     mae,
            "mape":    mape,
            "smape":   smape,
        })

    return pd.DataFrame(records)

