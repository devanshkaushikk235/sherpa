"""Table schemas + validators. Dates are UTC calendar days stored as midnight datetime64[ns]."""
import pandas as pd

SERIES_COLS = {
    "game_id": "object", "app_id": "int64", "date": "datetime64[ns]",
    "y_peak": "float64", "y_avg": "float64", "is_imputed": "bool", "is_outage": "bool",
}
PATCHES_COLS = {
    "game_id": "object", "patch_id": "object", "date": "datetime64[ns]", "title": "object",
    "magnitude": "float64", "patch_type": "object", "overlaps_prev": "bool",
}


def _check_cols(df: pd.DataFrame, spec: dict, name: str) -> None:
    missing = set(spec) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    for col, dtype in spec.items():
        dt = df[col].dtype
        if dtype == "object":          # pandas >=3 uses a dedicated str dtype; accept both
            ok = pd.api.types.is_string_dtype(dt) or dt == object
        elif dtype.startswith("datetime64"):   # resolution (ns/us) varies by pandas version
            ok = dt.kind == "M"
        elif dtype == "float64":
            ok = pd.api.types.is_numeric_dtype(dt)
        else:
            ok = str(dt) == dtype
        if not ok:
            raise ValueError(f"{name}.{col}: dtype {dt}, expected {dtype}")


def validate_series(df: pd.DataFrame) -> None:
    _check_cols(df, SERIES_COLS, "series")
    if df.duplicated(["game_id", "date"]).any():
        raise ValueError("series: duplicate (game_id, date)")
    for gid, g in df.groupby("game_id"):
        d = g.sort_values("date")["date"].diff().dropna()
        if not (d == pd.Timedelta(days=1)).all():
            raise ValueError(f"series[{gid}]: missing or irregular dates")
    bad = df["y_peak"].notna() & (df["y_peak"] <= 0)
    if bad.any():
        raise ValueError("series: non-positive y_peak present")


def validate_patches(df: pd.DataFrame) -> None:
    _check_cols(df, PATCHES_COLS, "patches")
    if df.duplicated(["game_id", "patch_id"]).any():
        raise ValueError("patches: duplicate patch_id")
    if not df["patch_type"].isin(["major", "minor"]).all():
        raise ValueError("patches: patch_type must be major|minor")


FORECASTS_COLS = {
    "game_id": "object", "origin_date": "datetime64[ns]", "horizon": "int64",
    "target_date": "datetime64[ns]", "model": "object", "y_hat": "float64", "y_true": "float64",
    "split": "object",
}


def validate_forecasts(df: pd.DataFrame) -> None:
    _check_cols(df, FORECASTS_COLS, "forecasts")
    if (df["horizon"] <= 0).any():
        raise ValueError("forecasts: horizon must be positive")
    diff = (df["target_date"] - df["origin_date"]).dt.days
    if not (diff == df["horizon"]).all():
        raise ValueError("forecasts: target_date must equal origin_date + horizon days")
    if df.duplicated(["game_id", "origin_date", "horizon", "model"]).any():
        raise ValueError("forecasts: duplicate (game_id, origin_date, horizon, model)")


INTERVALS_COLS = {
    "game_id": "object", "origin_date": "datetime64[ns]", "horizon": "int64",
    "target_date": "datetime64[ns]", "model": "object", "arm": "object",
    "level": "float64", "lo": "float64", "hi": "float64", "y_true": "float64",
    "covered": "bool", "width": "float64", "norm_width": "float64",
    "days_since_patch": "float64", "patch_type": "object", "in_patch_window": "bool",
    "split": "object", "patch_id": "object",
}


def validate_intervals(df: pd.DataFrame) -> None:
    _check_cols(df, INTERVALS_COLS, "intervals")
    if (df["lo"] > df["hi"]).any():
        raise ValueError("intervals: lo > hi detected")
    if (df["lo"] < 0).any():
        raise ValueError("intervals: negative lo bound detected")
    if not ((df["level"] > 0) & (df["level"] < 1)).all():
        raise ValueError("intervals: level must be in (0, 1)")
    if df.duplicated(["game_id", "origin_date", "horizon", "model", "arm", "level"]).any():
        raise ValueError("intervals: duplicate (game_id, origin_date, horizon, model, arm, level)")


METRICS_COLS = {
    "game_id": "object", "model": "object", "arm": "object", "level": "float64",
    "window": "object", "n": "int64", "coverage": "float64", "mean_width": "float64",
    "winkler": "float64", "mae": "float64", "smape": "float64",
}


def validate_metrics(df: pd.DataFrame) -> None:
    _check_cols(df, METRICS_COLS, "metrics")
    if not ((df["coverage"] >= 0) & (df["coverage"] <= 1)).all():
        raise ValueError("metrics: coverage must be in [0, 1]")
    if (df["winkler"] < 0).any():
        raise ValueError("metrics: winkler score cannot be negative")
