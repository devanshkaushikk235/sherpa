"""Synchronizes processed time-series, patches, intervals, and metrics into Supabase PostgreSQL.
Usage:
  python src/ingest/sync_supabase.py
"""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from supabase import create_client

from src.config import ROOT


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def sync_table_in_batches(client, table_name: str, df: pd.DataFrame, batch_size: int = 500, upsert_key: str = None):
    # Convert dates to ISO strings for JSON serialization
    records = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v) or (isinstance(v, (float, int)) and np.isinf(v)):
                clean_row[k] = None
            elif isinstance(v, pd.Timestamp):
                clean_row[k] = v.strftime("%Y-%m-%d")
            else:
                clean_row[k] = v
        records.append(clean_row)

    total = len(records)
    print(f"[{table_name}] Uploading {total} rows in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        if upsert_key:
            client.table(table_name).upsert(batch, on_conflict=upsert_key).execute()
        else:
            client.table(table_name).insert(batch).execute()
    print(f"[{table_name}] Successfully synced {total} rows!")


def main():
    load_env()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_KEY missing in .env")
        return

    client = create_client(url, key)
    print(f"Connected to Supabase project at {url}")

    interim_dir = ROOT / "data" / "interim"
    processed_dir = ROOT / "data" / "processed"

    # 1. Sync Metrics (EVAL split metrics, 432 rows)
    metrics_path = processed_dir / "metrics.parquet"
    if metrics_path.exists():
        m_df = pd.read_parquet(metrics_path)
        metric_cols = ["game_id", "model", "arm", "level", "window", "n", "coverage", "mean_width", "winkler", "mae", "smape"]
        m_df = m_df[[c for c in metric_cols if c in m_df.columns]]
        try:
            sync_table_in_batches(client, "liveops_metrics", m_df, batch_size=200, upsert_key="game_id,model,arm,level,window")
        except Exception as e:
            print(f"[liveops_metrics] Failed: {e}")

    # 2. Sync Capacity Sensitivity
    cap_path = processed_dir / "capacity_sensitivity.parquet"
    if cap_path.exists():
        c_df = pd.read_parquet(cap_path)[["game_id", "model", "arm", "level", "penalty_ratio", "norm_cost"]]
        try:
            sync_table_in_batches(client, "liveops_capacity_sensitivity", c_df, batch_size=200, upsert_key="game_id,model,arm,level,penalty_ratio")
        except Exception as e:
            print(f"[liveops_capacity_sensitivity] Failed: {e}")

    # 3. Sync Patches
    patches_path = interim_dir / "patches.parquet"
    if patches_path.exists():
        p_df = pd.read_parquet(patches_path)
        try:
            sync_table_in_batches(client, "liveops_patches", p_df, batch_size=200, upsert_key="game_id,patch_id")
        except Exception as e:
            print(f"[liveops_patches] Failed: {e}")

    # 4. Sync Series
    series_path = interim_dir / "series.parquet"
    if series_path.exists():
        s_df = pd.read_parquet(series_path)
        try:
            sync_table_in_batches(client, "liveops_series", s_df, batch_size=500, upsert_key="game_id,date")
        except Exception as e:
            print(f"[liveops_series] Failed: {e}")

    # 5. Sync Representative 1-day ahead intervals (EVAL split only)
    intervals_path = processed_dir / "intervals.parquet"
    if intervals_path.exists():
        i_df = pd.read_parquet(intervals_path)
        cols = [
            "game_id", "origin_date", "horizon", "target_date", "model", "arm",
            "level", "lo", "hi", "y_true", "covered", "width", "norm_width",
            "days_since_patch", "patch_type", "in_patch_window"
        ]
        # Filter strictly to EVAL split, h=1
        h1_df = i_df[(i_df["horizon"] == 1) & (i_df["split"] == "EVAL")][cols].copy()
        try:
            sync_table_in_batches(client, "liveops_intervals", h1_df, batch_size=500)
        except Exception as e:
            print(f"[liveops_intervals] Failed: {e}")

    print("\n=== Supabase Data Sync Completed! ===")


if __name__ == "__main__":
    main()
