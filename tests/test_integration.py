"""Integration test: full end-to-end pipeline on synthetic data.
Verifies the demo pipeline produces all expected output artefacts without errors.
This is the single most important CI gate — if run.py cmd_demo breaks, every
downstream evaluation metric becomes unreliable.
"""
import importlib
import sys
from pathlib import Path
import pandas as pd
import pytest

# Ensure project root is on the path (mirrors how run.py is invoked)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_full_demo_pipeline(tmp_path):
    """Runs the complete pipeline on synthetic data and checks all output artefacts exist."""
    from src.config import load_config
    import run as run_module

    cfg = load_config()
    # Fast cadence for CI integration test
    cfg["backtest"]["step_days"] = 35
    cfg.setdefault("eval", {})["bootstrap"] = {"n_resamples": 50}
    cfg.setdefault("lgbm", {})["n_estimators"] = 10

    from src.ingest import synthetic

    root = tmp_path
    games = synthetic.DEMO_GAMES
    for i, gid in enumerate(games):
        raw = root / "raw" / gid
        raw.mkdir(parents=True)
        daily = synthetic.make_daily(912, "2022-01-01", seed=i,
                                     shift_days=(200, 450, 700), decay=10)
        daily.to_csv(raw / "daily.csv", index=False)
        pd.DataFrame({
            "date":        ["2022-07-20", "2023-01-16", "2023-06-05", "2023-12-02"],
            "title":       ["Balance patch", "Major Season launch", "Expansion update", "Winter Major Season"],
            "force_major": [False, True, True, True],
        }).to_csv(raw / "patches_manual.csv", index=False)

    run_module.cmd_clean(games, cfg, root)
    run_module.cmd_backtest(games, cfg, root)
    run_module.cmd_eval(games, cfg, root)

    processed = root / "processed"
    for expected in [
        "forecasts.parquet",
        "intervals.parquet",
        "metrics.parquet",
        "patch_level.parquet",
        "bootstrap_results.parquet",
        "capacity_sensitivity.parquet",
        "point_forecast.parquet",
    ]:
        assert (processed / expected).exists(), f"Missing: {expected}"

    # Sanity-check metrics schema
    metrics = pd.read_parquet(processed / "metrics.parquet")
    assert "coverage" in metrics.columns
    assert "winkler"  in metrics.columns
    assert "norm_width" in metrics.columns
    assert not metrics.empty

    # Sanity-check point forecast table
    pt = pd.read_parquet(processed / "point_forecast.parquet")
    assert "mape" in pt.columns
    assert "window" in pt.columns
    assert not pt.empty
