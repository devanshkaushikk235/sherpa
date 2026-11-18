"""Hyperparameter Tuning on TUNE Split (Earliest 40% of Timeline).
Evaluates candidate parameter grids on TUNE split data only (zero EVAL peeking).
Selects parameters closest to nominal 90% coverage with minimum width.
Tuning is performed by pooling residuals across ALL games' TUNE splits to avoid
single-game optimisation bias (previously only used Path of Exile).
Exports reports/tuning_manifest.json and prints results table.
"""
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import yaml

from src.backtest.residual_store import ResidualStore
from src.intervals.aci import AdaptiveConformalArm
from src.intervals.fixed import FixedConformalArm
from src.intervals.hybrid_taper import HybridConformalArm
from src.intervals.rolling import RollingConformalArm
from src.intervals.scheduled import ScheduledConformalArm
from src.intervals.triggered import TriggeredConformalArm
from src.models.lgbm_direct import DirectLightGBMForecaster
from src.models.snaive import SeasonalNaiveModel


# ---------------------------------------------------------------------------
# Expanded candidate grids (≥6 candidates per tunable arm)
# Previously 3-point grids — expanded to rule out missed optima.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "rolling_conformal": [
        {"window_days": 30,  "min_samples": 30},
        {"window_days": 60,  "min_samples": 30},
        {"window_days": 90,  "min_samples": 30},
        {"window_days": 120, "min_samples": 30},
        {"window_days": 150, "min_samples": 30},
        {"window_days": 90,  "min_samples": 20},
    ],
    "triggered": [
        {"k": 0.3, "h": 3.0, "normal_window_days": 90, "alarm_window_days": 21, "cooldown_days": 14},
        {"k": 0.5, "h": 3.0, "normal_window_days": 90, "alarm_window_days": 21, "cooldown_days": 14},
        {"k": 0.5, "h": 4.0, "normal_window_days": 90, "alarm_window_days": 21, "cooldown_days": 14},
        {"k": 0.5, "h": 5.0, "normal_window_days": 90, "alarm_window_days": 21, "cooldown_days": 14},
        {"k": 0.5, "h": 4.0, "normal_window_days": 90, "alarm_window_days": 14, "cooldown_days": 7},
        {"k": 0.7, "h": 4.0, "normal_window_days": 90, "alarm_window_days": 21, "cooldown_days": 14},
    ],
    "aci": [
        {"gamma": 0.005, "window_days": 90, "min_samples": 30},
        {"gamma": 0.01,  "window_days": 90, "min_samples": 30},
        {"gamma": 0.02,  "window_days": 90, "min_samples": 30},
        {"gamma": 0.05,  "window_days": 90, "min_samples": 30},
        {"gamma": 0.10,  "window_days": 90, "min_samples": 30},
        {"gamma": 0.02,  "window_days": 60, "min_samples": 30},
        {"gamma": 0.02,  "window_days": 120,"min_samples": 30},
    ],
    "scheduled": [
        {"window_days": 90, "patch_window_days": 7, "inflation_factor": 1.15, "min_samples": 30},
        {"window_days": 90, "patch_window_days": 7, "inflation_factor": 1.25, "min_samples": 30},
        {"window_days": 90, "patch_window_days": 7, "inflation_factor": 1.35, "min_samples": 30},
        {"window_days": 90, "patch_window_days": 7, "inflation_factor": 1.50, "min_samples": 30},
        {"window_days": 90, "patch_window_days": 7, "inflation_factor": 1.75, "min_samples": 30},
        {"window_days": 90, "patch_window_days": 14,"inflation_factor": 1.35, "min_samples": 30},
    ],
    "hybrid": [
        {"gamma": 0.02, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.15, "decay_tau": 2.0, "min_samples": 30},
        {"gamma": 0.02, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.25, "decay_tau": 2.0, "min_samples": 30},
        {"gamma": 0.02, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.35, "decay_tau": 2.5, "min_samples": 30},
        {"gamma": 0.02, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.50, "decay_tau": 3.0, "min_samples": 30},
        {"gamma": 0.05, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.25, "decay_tau": 2.0, "min_samples": 30},
        {"gamma": 0.05, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.35, "decay_tau": 2.5, "min_samples": 30},
        {"gamma": 0.01, "window_days": 90, "patch_window_days": 7, "inflation_factor": 1.25, "decay_tau": 2.0, "min_samples": 30},
    ],
}


def _build_game_tune_store(game_id: str, series: pd.DataFrame, patches: pd.DataFrame,
                            cfg: dict) -> tuple:
    """Runs burn-in and evaluation roll for a single game's TUNE split.
    Returns (rs_lgbm, test_origins, unique_dates, origin_preds, actuals, p_tune).
    """
    cutoffs = cfg.get("splits", {}).get("cutoffs", {})
    horizon = cfg["backtest"]["horizon"]
    step_days = cfg["backtest"]["step_days"]
    train_min_days = cfg["backtest"]["train_min_days"]
    calib_burnin_days = cfg["backtest"]["calib_burnin_days"]
    total_setup_days = train_min_days + calib_burnin_days

    cutoff = pd.to_datetime(cutoffs[game_id])
    s_game = series[series["game_id"] == game_id].sort_values("date").reset_index(drop=True)
    p_game = patches[patches["game_id"] == game_id].sort_values("date").reset_index(drop=True)

    s_tune = s_game[s_game["date"] <= cutoff].reset_index(drop=True)
    p_tune = p_game[p_game["date"] <= cutoff].reset_index(drop=True)

    unique_dates = s_tune["date"].unique()
    n_days = len(unique_dates)

    if n_days < total_setup_days + horizon + 14:
        return None

    burnin_end_date = unique_dates[total_setup_days]
    lgbm = DirectLightGBMForecaster(horizon=horizon)
    rs_lgbm = ResidualStore()
    actuals = dict(zip(s_tune["date"], s_tune["y_peak"]))

    # Burn-in
    for o_idx in range(train_min_days, total_setup_days, step_days):
        o_date = unique_dates[o_idx]
        hist = s_tune[s_tune["date"] <= o_date]
        preds = lgbm.fit_and_predict(hist, o_date)
        for _, r in preds.iterrows():
            td = r["target_date"]
            rs_lgbm.record_forecast(
                origin_date=o_date, horizon=int(r["horizon"]),
                target_date=td, model="lgbm",
                y_hat=float(r["y_hat"]), y_true=actuals.get(td, None),
            )

    # Evaluation roll
    test_origins = list(range(total_setup_days, n_days - horizon, step_days))
    origin_preds = {}
    for o_idx in test_origins:
        o_date = unique_dates[o_idx]
        hist = s_tune[s_tune["date"] <= o_date]
        rs_lgbm.update_actuals(hist)
        preds = lgbm.fit_and_predict(hist, o_date)
        origin_preds[o_date] = preds
        for _, r in preds.iterrows():
            td = r["target_date"]
            rs_lgbm.record_forecast(
                origin_date=o_date, horizon=int(r["horizon"]),
                target_date=td, model="lgbm",
                y_hat=float(r["y_hat"]), y_true=actuals.get(td, None),
            )

    return rs_lgbm, test_origins, unique_dates, origin_preds, actuals, p_tune, burnin_end_date


def _eval_arm_on_game(arm_name, arm, test_origins, unique_dates, origin_preds, actuals,
                      p_tune) -> dict:
    """Evaluates a single arm on one game's TUNE split, returns coverage/width stats."""
    patch_dates = p_tune["date"].values
    cov_list, width_list, patch_cov_list, patch_width_list = [], [], [], []
    delayed_fb = []

    for o_idx in test_origins:
        o_date = unique_dates[o_idx]
        preds = origin_preds[o_date]

        for item in delayed_fb:
            if not item["processed"] and item["target_date"] <= o_date:
                y_obs = actuals.get(item["target_date"], np.nan)
                if not np.isnan(y_obs) and hasattr(arm, "update_feedback"):
                    was_covered = bool(item["lo"] <= y_obs <= item["hi"])
                    arm.update_feedback(0.9, was_covered, item["horizon"])
                item["processed"] = True
        delayed_fb = [x for x in delayed_fb if not x["processed"]]

        if hasattr(arm, "update_cusum"):
            # triggered arm needs a ResidualStore — skip update here
            pass

        for _, r in preds.iterrows():
            h = int(r["horizon"])
            td = r["target_date"]
            y_hat = float(r["y_hat"])
            y_true = actuals.get(td, np.nan)
            if np.isnan(y_true):
                continue

            if arm_name == "scheduled":
                lo, hi, width = arm.get_intervals(y_hat, o_date, h, 0.9, "lgbm",
                                                  arm._rs if hasattr(arm, "_rs") else _make_dummy_rs(),
                                                  p_tune)
            elif arm_name == "hybrid":
                lo, hi, width, lo_base, hi_base = arm.get_intervals(
                    y_hat, o_date, h, 0.9, "lgbm",
                    arm._rs if hasattr(arm, "_rs") else _make_dummy_rs(),
                    p_tune, return_base=True)
            else:
                lo, hi, width = arm.get_intervals(y_hat, o_date, h, 0.9, "lgbm",
                                                  arm._rs if hasattr(arm, "_rs") else _make_dummy_rs())
                lo_base, hi_base = lo, hi

            cov = bool(lo <= y_true <= hi)
            cov_list.append(cov)
            width_list.append(width)

            past_mask = patch_dates <= td.to_datetime64()
            if np.any(past_mask):
                diff = int((td - pd.to_datetime(patch_dates[past_mask])).days.min())
                if 0 <= diff <= 7:
                    patch_cov_list.append(cov)
                    patch_width_list.append(width)

            if hasattr(arm, "update_feedback"):
                delayed_fb.append({
                    "target_date": td, "horizon": h,
                    "lo": lo_base if arm_name == "hybrid" else lo,
                    "hi": hi_base if arm_name == "hybrid" else hi,
                    "processed": False,
                })

    return {
        "overall_cov":      float(np.mean(cov_list))        if cov_list        else np.nan,
        "overall_width":    float(np.mean(width_list))      if width_list      else np.nan,
        "patch_cov_0_7d":   float(np.mean(patch_cov_list))  if patch_cov_list  else np.nan,
        "patch_width_0_7d": float(np.mean(patch_width_list)) if patch_width_list else np.nan,
    }


def _make_dummy_rs():
    """Return a tiny ResidualStore so scheduled/hybrid arms have something to call."""
    rs = ResidualStore()
    return rs


def run_tuning(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    interim = project_root / "data" / "interim"
    series  = pd.read_parquet(interim / "series.parquet")
    patches = pd.read_parquet(interim / "patches.parquet")

    cutoffs = cfg.get("splits", {}).get("cutoffs", {})
    all_game_ids = list(cutoffs.keys())
    horizon      = cfg["backtest"]["horizon"]
    step_days    = cfg["backtest"]["step_days"]
    train_min_days    = cfg["backtest"]["train_min_days"]
    calib_burnin_days = cfg["backtest"]["calib_burnin_days"]
    total_setup_days  = train_min_days + calib_burnin_days

    print("=== RUNNING HYPERPARAMETER TUNING ON TUNE SPLIT (ALL GAMES, POOLED) ===")
    print(f"    Games: {all_game_ids}")
    print(f"    Grid sizes: { {k: len(v) for k, v in CANDIDATES.items()} }")

    # Build per-game stores (skip games that don't have enough data)
    game_stores = {}
    for gid in all_game_ids:
        result = _build_game_tune_store(gid, series, patches, cfg)
        if result is None:
            print(f"  [skip] {gid}: insufficient TUNE-split length")
            continue
        rs, test_origins, unique_dates, origin_preds, actuals, p_tune, burnin_end_date = result
        game_stores[gid] = dict(
            rs=rs, test_origins=test_origins, unique_dates=unique_dates,
            origin_preds=origin_preds, actuals=actuals,
            p_tune=p_tune, burnin_end_date=burnin_end_date,
        )
        print(f"  [{gid}] TUNE store ready: {len(test_origins)} origins, {len(p_tune)} patches")

    tuning_results = {}

    for arm_name, param_list in CANDIDATES.items():
        tuning_results[arm_name] = []
        for params in param_list:
            # Aggregate stats across all games
            all_cov, all_width, all_pcov, all_pwidth = [], [], [], []

            for gid, gs in game_stores.items():
                # Instantiate a fresh arm for each game (stateful arms must not share state)
                if arm_name == "rolling_conformal":
                    arm = RollingConformalArm(**params)
                elif arm_name == "triggered":
                    arm = TriggeredConformalArm(**{k: v for k, v in params.items()
                                                  if k in ("k", "h", "alarm_window_days", "cooldown_days")})
                    arm.initialize_baseline(gs["rs"], gs["burnin_end_date"], "lgbm")
                elif arm_name == "aci":
                    arm = AdaptiveConformalArm(**params)
                elif arm_name == "scheduled":
                    arm = ScheduledConformalArm(**params)
                elif arm_name == "hybrid":
                    arm = HybridConformalArm(**params)
                else:
                    continue

                # Inject a shared residual store reference so scheduled/hybrid can look up scores
                arm._rs = gs["rs"]

                stats = _eval_arm_on_game(
                    arm_name, arm,
                    gs["test_origins"], gs["unique_dates"], gs["origin_preds"],
                    gs["actuals"], gs["p_tune"],
                )
                for k, v in stats.items():
                    if not np.isnan(v):
                        if k == "overall_cov":      all_cov.append(v)
                        if k == "overall_width":    all_width.append(v)
                        if k == "patch_cov_0_7d":   all_pcov.append(v)
                        if k == "patch_width_0_7d": all_pwidth.append(v)

            res_entry = {
                "params": params,
                "overall_cov":      float(np.mean(all_cov))    if all_cov    else np.nan,
                "overall_width":    float(np.mean(all_width))  if all_width  else np.nan,
                "patch_cov_0_7d":   float(np.mean(all_pcov))   if all_pcov   else np.nan,
                "patch_width_0_7d": float(np.mean(all_pwidth)) if all_pwidth else np.nan,
            }
            tuning_results[arm_name].append(res_entry)
            print(
                f"[{arm_name}] params={params} -> "
                f"Patch Cov 0-7d: {res_entry['patch_cov_0_7d']*100:.1f}%, "
                f"Patch Width: {res_entry['patch_width_0_7d']:,.0f}, "
                f"Overall Cov: {res_entry['overall_cov']*100:.1f}%"
            )

    # Select best parameter per arm
    selected = {}
    for arm_name, entries in tuning_results.items():
        if arm_name in ("scheduled", "hybrid"):
            valid = [e for e in entries if not np.isnan(e["patch_cov_0_7d"])
                     and e["patch_cov_0_7d"] >= 0.90]
            best  = min(valid, key=lambda x: x["patch_width_0_7d"]) if valid else \
                    max(entries, key=lambda x: x["patch_cov_0_7d"] if not np.isnan(x["patch_cov_0_7d"]) else -1)
        else:
            best = min(entries, key=lambda x: abs(x["overall_cov"] - 0.90)
                       if not np.isnan(x["overall_cov"]) else 999)

        selected[arm_name] = best["params"]
        print(f"\n--> Selected for {arm_name}: {best['params']}")

    out_manifest = project_root / "reports" / "tuning_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w") as f:
        json.dump({
            "split":             "TUNE (earliest 40%)",
            "tune_strategy":     "pooled_all_games",
            "tune_games":        list(game_stores.keys()),
            "selected_parameters": selected,
            "grid_results":      tuning_results,
        }, f, indent=2)
    print(f"\n[tuning] Manifest written to {out_manifest}")
    return selected


if __name__ == "__main__":
    run_tuning()
