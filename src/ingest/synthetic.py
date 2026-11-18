"""Synthetic stand-in data for offline demo + tests. Weekly seasonality, noise, level shifts at patches."""
import numpy as np
import pandas as pd

DEMO_GAMES = {
    "demo_a": {"name": "Demo A", "app_id": 1, "start": "2022-01-01", "end": "2024-06-30"},
    "demo_b": {"name": "Demo B", "app_id": 2, "start": "2022-01-01", "end": "2024-06-30"},
}


def make_daily(n_days=900, start="2022-01-01", seed=0, shift_days=(), shift_size=0.4, decay=14, base=50_000):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="D")
    dow = np.array([1.0, 1.0, 1.0, 1.02, 1.08, 1.2, 1.15])[idx.dayofweek]
    level = np.full(n_days, float(base))
    for d in shift_days:
        t = np.arange(n_days) - d
        level *= 1 + np.where(t >= 0, shift_size * np.exp(-np.clip(t, 0, None) / decay), 0)
    y = level * dow * rng.lognormal(0, 0.03, n_days)
    return pd.DataFrame({"date": idx, "y_peak": y, "y_avg": y * 0.8})


def make_patch_candidates(game_id, dates, titles=None):
    return pd.DataFrame({
        "game_id": game_id,
        "date": pd.to_datetime(list(dates)),
        "title": titles or [f"Update {i}" for i in range(len(dates))],
        "force_major": False,
    })
