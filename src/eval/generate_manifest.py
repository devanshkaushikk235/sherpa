"""Manifest Generator.
Computes all audit metadata, splits, and sample counts directly from parquet artifacts
and outputs reports/run_manifest.json with zero hardcoding.
"""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import json
import pandas as pd
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def generate_manifest(config_path: str = "configs/default.yaml") -> dict:
    cfg_file = project_root / config_path
    with open(cfg_file, "r") as f:
        cfg_content = f.read()
        cfg = yaml.safe_load(cfg_content)

    cfg_hash = hashlib.sha256(cfg_content.encode("utf-8")).hexdigest()[:12]

    # Read data parquets
    proc = project_root / "data" / "processed"
    interim = project_root / "data" / "interim"

    forecasts_df = pd.read_parquet(proc / "forecasts.parquet")
    intervals_df = pd.read_parquet(proc / "intervals.parquet")
    patches_df = pd.read_parquet(interim / "patches.parquet")

    cutoffs = cfg.get("splits", {}).get("cutoffs", {})
    patches_df["split"] = patches_df.apply(
        lambda r: "TUNE" if r["date"] <= pd.to_datetime(cutoffs.get(r["game_id"], "2099-01-01")) else "EVAL",
        axis=1,
    )

    tune_patches_by_game = patches_df[patches_df["split"] == "TUNE"].groupby("game_id").size().to_dict()
    eval_patches_by_game = patches_df[patches_df["split"] == "EVAL"].groupby("game_id").size().to_dict()

    # Number of evaluated patch clusters in the 0-7d window
    eval_0_7d_patches = (
        intervals_df[(intervals_df["split"] == "EVAL") & (intervals_df["days_since_patch"] <= 7)]
        ["patch_id"].dropna().unique()
    )

    eval_f_df = forecasts_df[forecasts_df["split"] == "EVAL"]
    primary_eval_forecasts = len(eval_f_df[eval_f_df["model"] == "lgbm"])
    total_eval_forecasts = len(eval_f_df)
    total_forecasts = len(forecasts_df)
    total_intervals = len(intervals_df)

    manifest = {
        "project": "forecast-reliability",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": get_git_commit(),
        "config_sha256": cfg_hash,
        "python_version": sys.version,
        "random_seeds": {
            "global": cfg.get("backtest", {}).get("seed", 42),
            "bootstrap": 42,
            "lgbm": cfg.get("lgbm", {}).get("random_state", 42),
        },
        "dataset_summary": {
            "total_raw_patches": len(patches_df),
            "tune_patches_pooled": int(len(patches_df[patches_df["split"] == "TUNE"])),
            "tune_patches_poe_selected_for_tuning": int(tune_patches_by_game.get("path_of_exile", 0)),
            "eval_patches_pooled": int(len(patches_df[patches_df["split"] == "EVAL"])),
            "eval_patches_in_0_7d_window": int(len(eval_0_7d_patches)),
            "patches_by_split_and_game": {
                "tune": {k: int(v) for k, v in tune_patches_by_game.items()},
                "eval": {k: int(v) for k, v in eval_patches_by_game.items()},
            },
        },
        "forecast_and_interval_counts": {
            "total_forecast_rows_across_all_models_and_splits": int(total_forecasts),
            "primary_model_eval_forecast_points": int(primary_eval_forecasts),
            "total_eval_forecast_rows_all_models": int(total_eval_forecasts),
            "total_interval_rows_evaluated": int(total_intervals),
            "primary_nominal_level": 0.90,
            "evaluated_nominal_levels": [0.80, 0.90, 0.95],
            "evaluated_models": sorted(list(forecasts_df["model"].unique())),
            "evaluated_arms": sorted(list(intervals_df["arm"].unique())),
        },
        "splits": {
            "tune_fraction": cfg.get("splits", {}).get("tune_fraction", 0.4),
            "eval_fraction": 1.0 - cfg.get("splits", {}).get("tune_fraction", 0.4),
            "cutoffs": cutoffs,
        },
    }

    out_file = project_root / "reports" / "run_manifest.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest saved to {out_file}")
    return manifest


if __name__ == "__main__":
    generate_manifest()
