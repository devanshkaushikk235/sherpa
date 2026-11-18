import numpy as np
import pandas as pd


def _fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Linear-interpolate NaN runs of length <= max_gap that sit between observed values."""
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_len <= max_gap)
    interp = s.interpolate(method="linear", limit_area="inside")
    filled = s.where(~fillable, interp)
    return filled, fillable & filled.notna()


def clean_series(daily: pd.DataFrame, game_id: str, app_id: int, start=None, end=None,
                 max_gap_days: int = 3, drop_ratio: float = 0.2, drop_window: int = 15) -> pd.DataFrame:
    df = daily.sort_values("date").drop_duplicates("date").set_index("date")
    lo = pd.Timestamp(start) if start else df.index.min()
    hi = pd.Timestamp(end) if end else df.index.max()
    df = df.reindex(pd.date_range(lo, hi, freq="D"))
    df.index.name = "date"
    for col in ("y_peak", "y_avg"):
        if col not in df:
            df[col] = np.nan

    peak = df["y_peak"].where(df["y_peak"] > 0)
    med = peak.rolling(drop_window, center=True, min_periods=5).median()
    flagged = (df["y_peak"] <= 0) | (peak < drop_ratio * med)   # zeros and outage dips; never upward spikes
    flagged = flagged.fillna(False)

    cols = {}
    for col in ("y_peak", "y_avg"):
        s = df[col].where((df[col] > 0) & ~flagged)
        cols[col], imputed = _fill_short_gaps(s, max_gap_days)
        if col == "y_peak":
            is_imputed = imputed

    out = pd.DataFrame({
        "game_id": game_id,
        "app_id": np.int64(app_id),
        "date": df.index,
        "y_peak": cols["y_peak"].to_numpy(dtype="float64"),
        "y_avg": cols["y_avg"].to_numpy(dtype="float64"),
        "is_imputed": is_imputed.to_numpy(dtype=bool),
        "is_outage": (flagged | cols["y_peak"].isna()).to_numpy(dtype=bool),  # True => y_peak flagged or unusable
    })
    out["app_id"] = out["app_id"].astype("int64")
    return out.reset_index(drop=True)
