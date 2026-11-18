"""Single CLI entry point for data ingestion, cleaning, backtesting, evaluation, and figures.
Usage:
  python run.py {ingest|check|clean|backtest|eval|figures|demo|all} [--game G] [--data-root DIR]
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import schemas
from src.backtest.runner import run_backtest_for_game
from src.clean.patch_tagger import tag_patches
from src.clean.series_cleaner import clean_series
from src.config import ROOT, load_config, load_games
from src.eval.bootstrap import block_bootstrap_paired_diff
from src.eval.capacity_cost import (
    compute_capacity_cost_sensitivity,
    compute_break_even_penalty_ratio,
)
from src.eval.cluster_bootstrap import run_cluster_bootstrap
from src.eval.patch_level import aggregate_patch_level
from src.eval.windows import aggregate_strata_metrics, point_forecast_table
from src.ingest import steam_news as sn
from src.ingest import steamcharts as sc
from src.ingest import synthetic
from src.viz import figures as viz


def select(games: dict, only: str | None) -> dict:
    return {only: games[only]} if only else games


def cmd_ingest(games, cfg, data_root):
    ic = cfg["ingest"]
    for gid, g in games.items():
        raw = data_root / "raw" / gid
        print(f"[ingest] {gid} (app {g['app_id']})")
        for label, fn in (
            ("steamcharts", lambda: sc.fetch_raw(g["app_id"], raw, ic["sleep_seconds"])),
            ("steam_news", lambda: sn.fetch_news(g["app_id"], raw, ic["news_count"],
                                                 ic["news_max_pages"], ic["sleep_seconds"])),
        ):
            try:
                print(f"  {label}: {fn()}")
            except Exception as e:
                print(f"  {label}: FAILED ({type(e).__name__}: {e})")
                print(f"    -> place data manually in {raw} (see README)")


def cmd_check(games, cfg, data_root):
    for gid in games:
        raw = data_root / "raw" / gid
        js = raw / "steamcharts_chart.json"
        if not js.exists():
            print(f"[check] {gid}: no steamcharts_chart.json (daily.csv fallback?)")
            continue
        rep = sc.sampling_report(sc.parse_chart_json(js))
        print(f"[check] {gid}: {rep}")
        if rep["median_obs_per_day"] < 4:
            print("  WARNING: <4 obs/day: y_peak undercounts the true peak, y_avg is weak. Consider another source.")


def cmd_clean(games, cfg, data_root):
    cc, pc = cfg["clean"], cfg["patches"]
    series, cands = [], []
    for gid, g in games.items():
        raw = data_root / "raw" / gid
        s = clean_series(sc.load_daily(raw), gid, g["app_id"], g.get("start"), g.get("end"),
                         cc["max_gap_days"], cc["drop_ratio"], cc["drop_window"])
        series.append(s)
        cands.append(sn.load_patch_candidates(raw, gid, g.get("patch_regex", pc["patch_regex"]), pc["feeds"]))
    series = pd.concat(series, ignore_index=True)
    schemas.validate_series(series)
    patches = tag_patches(pd.concat(cands, ignore_index=True), series, pc["window_days"],
                          pc["major_threshold"], pc["overlap_days"], pc["major_regex"])
    schemas.validate_patches(patches)
    out = data_root / "interim"
    out.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out / "series.parquet", index=False)
    patches.to_parquet(out / "patches.parquet", index=False)
    for gid, s in series.groupby("game_id"):
        p = patches[patches["game_id"] == gid]
        print(f"[clean] {gid}: {len(s)} days {s['date'].min().date()}..{s['date'].max().date()} | "
              f"imputed {int(s['is_imputed'].sum())} | unusable/outage {int(s['is_outage'].sum())} | "
              f"patches {len(p)} (major {int((p['patch_type'] == 'major').sum())}, "
              f"overlapping {int(p['overlaps_prev'].sum())})")
    print(f"[clean] wrote {out}/series.parquet, patches.parquet")


def cmd_backtest(games, cfg, data_root):
    interim = data_root / "interim"
    series_path = interim / "series.parquet"
    patches_path = interim / "patches.parquet"
    if not series_path.exists() or not patches_path.exists():
        raise FileNotFoundError(f"Missing interim data in {interim}. Run 'clean' first.")

    series = pd.read_parquet(series_path)
    patches = pd.read_parquet(patches_path)

    all_forecasts, all_intervals = [], []
    for gid in games:
        print(f"\n--- [backtest] Running backtest for {gid} ---")
        f_df, i_df = run_backtest_for_game(gid, series, patches, cfg)
        all_forecasts.append(f_df)
        all_intervals.append(i_df)

    f_combined = pd.concat(all_forecasts, ignore_index=True)
    i_combined = pd.concat(all_intervals, ignore_index=True)

    out = data_root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    f_combined.to_parquet(out / "forecasts.parquet", index=False)
    i_combined.to_parquet(out / "intervals.parquet", index=False)
    print(f"\n[backtest] Completed! Saved {len(f_combined)} forecasts and {len(i_combined)} intervals to {out}/")


def cmd_eval(games, cfg, data_root):
    processed = data_root / "processed"
    i_path = processed / "intervals.parquet"
    f_path = processed / "forecasts.parquet"
    if not i_path.exists() or not f_path.exists():
        raise FileNotFoundError(f"Missing processed data in {processed}. Run 'backtest' first.")

    intervals = pd.read_parquet(i_path)
    forecasts = pd.read_parquet(f_path)

    # Filter to requested games
    intervals = intervals[intervals["game_id"].isin(games)]
    forecasts = forecasts[forecasts["game_id"].isin(games)]

    # Restrict formal evaluation strictly to EVAL split if split column exists and contains EVAL rows
    has_eval_split = "split" in intervals.columns and (intervals["split"] == "EVAL").any()
    eval_intervals = (
        intervals[intervals["split"] == "EVAL"].copy()
        if has_eval_split
        else intervals.copy()
    )
    eval_forecasts = (
        forecasts[forecasts["split"] == "EVAL"].copy()
        if has_eval_split and "split" in forecasts.columns
        else forecasts.copy()
    )

    print("[eval] Computing window strata aggregations (EVAL split only)...")
    metrics_df = aggregate_strata_metrics(eval_intervals, eval_forecasts)
    schemas.validate_metrics(metrics_df)
    metrics_df.to_parquet(processed / "metrics.parquet", index=False)

    print("[eval] Generating patch-level aggregation table (0-7d shock window)...")
    ratios = cfg.get("eval", {}).get("capacity_cost", {}).get("ratios", [2.0, 5.0, 10.0])
    patch_level_df = aggregate_patch_level(intervals, ratios=ratios)
    patch_level_df.to_parquet(processed / "patch_level.parquet", index=False)
    print(f"[eval] Saved patch-level table with {len(patch_level_df)} records to {processed}/patch_level.parquet")

    print("[eval] Computing stratified cluster bootstrap over patch events...")
    b_cfg = cfg.get("eval", {}).get("bootstrap", {"n_resamples": 2000})
    cluster_boot_df = run_cluster_bootstrap(
        patch_level_df, n_resamples=b_cfg.get("n_resamples", 2000), seed=42
    )
    cluster_boot_df.to_parquet(processed / "bootstrap_results.parquet", index=False)

    print("[eval] Computing asymmetric capacity cost sensitivity and break-even penalty ratios...")
    cost_records = []
    for (gid, m, arm, lvl), g in eval_intervals.groupby(["game_id", "model", "arm", "level"]):
        sens = compute_capacity_cost_sensitivity(g, ratios=ratios)
        # Compute break-even ratio vs Fixed baseline
        base_g = eval_intervals[
            (eval_intervals["game_id"] == gid) &
            (eval_intervals["model"] == m) &
            (eval_intervals["arm"] == "fixed") &
            (eval_intervals["level"] == lvl)
        ]
        merged_break = pd.merge(
            g[["target_date", "horizon", "hi", "y_true"]],
            base_g[["target_date", "horizon", "hi"]],
            on=["target_date", "horizon"],
            suffixes=("_arm", "_base"),
        ).dropna()
        r_star = compute_break_even_penalty_ratio(
            merged_break["y_true"].values,
            merged_break["hi_arm"].values,
            merged_break["hi_base"].values,
        )

        for r, cost_val in sens.items():
            cost_records.append({
                "game_id": gid,
                "model": m,
                "arm": arm,
                "level": lvl,
                "penalty_ratio": r,
                "norm_cost": cost_val,
                "break_even_r_star": r_star,
            })
    cost_df = pd.DataFrame(cost_records)
    cost_df.to_parquet(processed / "capacity_sensitivity.parquet", index=False)

    print(f"[eval] Metrics saved to {processed}/ (metrics.parquet, patch_level.parquet, "
          f"bootstrap_results.parquet, capacity_sensitivity.parquet)")

    print("[eval] Computing standalone point forecast table by window...")
    pt_df = point_forecast_table(eval_forecasts, eval_intervals)
    pt_df.to_parquet(processed / "point_forecast.parquet", index=False)
    print(f"[eval] Saved point_forecast.parquet ({len(pt_df)} rows)")


def cmd_figures(games, cfg, data_root):
    interim = data_root / "interim"
    processed = data_root / "processed"
    fig_dir = ROOT / "reports" / "figures"

    series = pd.read_parquet(interim / "series.parquet")
    patches = pd.read_parquet(interim / "patches.parquet")
    intervals = pd.read_parquet(processed / "intervals.parquet")
    metrics = pd.read_parquet(processed / "metrics.parquet")

    # Filter to requested games
    series = series[series["game_id"].isin(games)]
    patches = patches[patches["game_id"].isin(games)]
    intervals = intervals[intervals["game_id"].isin(games)]
    metrics = metrics[metrics["game_id"].isin(games)]

    sens_path = processed / "ablation_sensitivity_kappa_tau.parquet"
    sens_df = pd.read_parquet(sens_path) if sens_path.exists() else pd.DataFrame()

    print(f"[figures] Generating 8 publication figures in {fig_dir}/ ...")
    viz.plot_fig1_coverage_by_window(metrics, fig_dir / "fig1_coverage_by_window_ci.png", level=0.9)
    viz.plot_fig2_width_coverage_frontier(metrics, fig_dir / "fig2_width_coverage_frontier.png", level=0.9)
    viz.plot_fig3_cost_sensitivity_breakeven(intervals, fig_dir / "fig3_cost_sensitivity_breakeven.png", level=0.9)
    viz.plot_fig4_event_study(intervals, fig_dir / "fig4_event_study_days_since_patch.png", level=0.9)
    viz.plot_fig5_sensitivity_heatmap(sens_df, fig_dir / "fig5_hybrid_sensitivity_heatmap.png")
    viz.plot_fig6_per_game_small_multiples(metrics, fig_dir / "fig6_per_game_small_multiples.png", level=0.9)
    viz.plot_fig7_reliability_diagram(metrics, fig_dir / "fig7_reliability_diagram.png")
    pt_path = processed / "point_forecast.parquet"
    if pt_path.exists():
        pt_df = pd.read_parquet(pt_path)
        viz.plot_fig8_point_forecast_degradation(pt_df, fig_dir / "fig8_point_forecast_degradation.png")
    print(f"[figures] Rendered 8 publication figures to {fig_dir}/")



def cmd_snowflake_load(data_root):
    try:
        from src.ingest.snowflake_loader import ingest_all_raw_data
        print("[snowflake] Loading raw JSON payloads into Snowflake RAW schema...")
        counts = ingest_all_raw_data(data_root)
        print(f"[snowflake] Ingestion complete: {counts}")
    except Exception as e:
        print(f"[snowflake] FAILED ({type(e).__name__}: {e})")
        print("  Ensure SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, etc. are set in your environment.")


def cmd_demo(cfg):
    print("=== RUNNING FAST END-TO-END DEMO ON SYNTHETIC SHIFT DATA ===")
    root = Path(tempfile.mkdtemp(prefix="fr_demo_"))
    games = synthetic.DEMO_GAMES
    for i, gid in enumerate(games):
        raw = root / "raw" / gid
        raw.mkdir(parents=True)
        # Generate 912 days of synthetic series matching DEMO_GAMES 2022-01-01..2024-06-30
        daily = synthetic.make_daily(912, "2022-01-01", seed=i, shift_days=(200, 450, 700), decay=10)
        daily.loc[150:151, "y_peak"] = float("nan")
        daily.to_csv(raw / "daily.csv", index=False)
        pd.DataFrame({
            "date": ["2022-07-20", "2023-01-16", "2023-06-05", "2023-12-02"],
            "title": ["Balance patch", "Major Season launch", "Expansion update", "Winter Major Season"],
            "force_major": [False, True, True, True],
        }).to_csv(raw / "patches_manual.csv", index=False)

    cmd_clean(games, cfg, root)
    cmd_backtest(games, cfg, root)
    cmd_eval(games, cfg, root)
    cmd_figures(games, cfg, root)
    print(f"\n[demo] Full pipeline validated successfully! Demo outputs in {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "check", "clean", "backtest", "eval", "figures", "demo", "snowflake-load", "all"])
    ap.add_argument("--game")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    cfg = load_config()
    if a.cmd == "demo":
        return cmd_demo(cfg)
    if a.cmd == "snowflake-load":
        return cmd_snowflake_load(a.data_root)
    games = select(load_games(), a.game)
    if a.cmd in ("ingest", "all"):
        cmd_ingest(games, cfg, a.data_root)
    if a.cmd in ("check", "all"):
        cmd_check(games, cfg, a.data_root)
    if a.cmd in ("clean", "all"):
        cmd_clean(games, cfg, a.data_root)
    if a.cmd in ("backtest", "all"):
        cmd_backtest(games, cfg, a.data_root)
    if a.cmd in ("eval", "all"):
        cmd_eval(games, cfg, a.data_root)
    if a.cmd in ("figures", "all"):
        cmd_figures(games, cfg, a.data_root)


if __name__ == "__main__":
    main()
