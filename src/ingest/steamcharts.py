"""SteamCharts telemetry ingestion module.
Loads sub-daily player concurrency data and aggregates to daily metrics (y_peak, y_avg).
Fallback: load from data/raw/<game_id>/daily.csv."""
import json
import time
from pathlib import Path

import pandas as pd
import requests

UA = "liveops-telemetry-engine/1.0 (LiveOps Telemetry Client)"
URL = "https://steamcharts.com/app/{app_id}/chart-data.json"


def fetch_raw(app_id: int, raw_dir: Path, sleep: float = 2.0, timeout: int = 30) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(URL.format(app_id=app_id), headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or not data:
        raise ValueError("unexpected chart-data.json shape")
    path = raw_dir / "steamcharts_chart.json"
    path.write_text(json.dumps(data))
    time.sleep(sleep)
    return path


def parse_chart_json(path: Path) -> pd.DataFrame:
    """[[ms_epoch, players], ...] -> DataFrame[ts (UTC), players]."""
    data = json.loads(Path(path).read_text())
    df = pd.DataFrame(data, columns=["ts_ms", "players"]).dropna()
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["ts", "players"]].sort_values("ts").reset_index(drop=True)


def sampling_report(df: pd.DataFrame) -> dict:
    spacing = df["ts"].diff().dropna()
    per_day = df.groupby(df["ts"].dt.floor("D")).size()
    return {
        "n_points": int(len(df)),
        "first": str(df["ts"].min().date()),
        "last": str(df["ts"].max().date()),
        "median_spacing_hours": float(spacing.median() / pd.Timedelta(hours=1)),
        "median_obs_per_day": float(per_day.median()),
        "share_days_with_ge4_obs": float((per_day >= 4).mean()),
    }


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sub-daily samples to UTC days. y_peak = max sample, y_avg = mean sample.
    If obs/day is low, y_peak undercounts the true peak; that is a data limitation, not a bug."""
    day = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    g = df.groupby(day)["players"]
    out = pd.DataFrame({"y_peak": g.max(), "y_avg": g.mean(), "n_obs": g.size()})
    out.index.name = "date"
    return out.reset_index()


def load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    if "y_avg" not in df.columns:
        df["y_avg"] = float("nan")
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize() if df["date"].dt.tz else df["date"].dt.normalize()
    return df[["date", "y_peak", "y_avg"]]


def load_daily(raw_dir: Path) -> pd.DataFrame:
    csv = raw_dir / "daily.csv"
    js = raw_dir / "steamcharts_chart.json"
    if csv.exists():
        return load_daily_csv(csv)
    if js.exists():
        return to_daily(parse_chart_json(js))
    raise FileNotFoundError(
        f"No daily data in {raw_dir}. Run `python run.py ingest` or place daily.csv (date,y_peak[,y_avg])."
    )
