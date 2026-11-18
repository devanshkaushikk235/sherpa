"""Rolling-Origin Backtest Runner.
Executes leakage-guarded rolling origin evaluation across games, models, and interval arms.
Phase 1: Initial training (days 1..365)
Phase 2: Burn-in out-of-sample calibration (days 366..455) to calibrate Arm A, CUSUM baseline, and initial ACI state
Phase 3: Formal evaluation scoring (days 456+) across all arms under identical conditions with delayed feedback.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.backtest.residual_store import ResidualStore
from src.intervals.aci import AdaptiveConformalArm
from src.intervals.fixed import FixedConformalArm
from src.intervals.hybrid_taper import HybridConformalArm
from src.intervals.rolling import RollingConformalArm
from src.intervals.scheduled import ScheduledConformalArm
from src.intervals.triggered import TriggeredConformalArm
from src.models.lgbm_direct import DirectLightGBMForecaster
from src.models.snaive import SeasonalNaiveModel
from src import schemas


def annotate_patch_context(df: pd.DataFrame, patches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotates each forecast/interval row with days_since_patch, patch_type, patch_id,
    and in_patch_window using a vectorised pd.merge_asof (O(n log n) vs O(n*p) row-loop).
    """
    if patches_df.empty or "date" not in patches_df.columns:
        df = df.copy()
        df["days_since_patch"] = np.nan
        df["patch_type"] = "none"
        df["patch_id"] = "none"
        df["in_patch_window"] = False
        return df

    p_cols = ["date", "patch_type"]
    if "patch_id" in patches_df.columns:
        p_cols.append("patch_id")
    p_sorted = (
        patches_df[p_cols]
        .sort_values("date")
        .rename(columns={"date": "_patch_date", "patch_type": "_patch_type"})
    )
    if "patch_id" in patches_df.columns:
        p_sorted = p_sorted.rename(columns={"patch_id": "_patch_id"})
    else:
        p_sorted["_patch_id"] = [
            "p_" + str(i) for i in range(len(p_sorted))
        ]

    orig_index = df.index
    df_sorted = df.sort_values("target_date").copy()
    df_sorted["target_date"] = pd.to_datetime(df_sorted["target_date"]).astype("datetime64[us]")
    p_sorted["_patch_date"] = pd.to_datetime(p_sorted["_patch_date"]).astype("datetime64[us]")

    merged = pd.merge_asof(
        df_sorted,
        p_sorted,
        left_on="target_date",
        right_on="_patch_date",
        direction="backward",
    )

    merged["days_since_patch"] = (
        (merged["target_date"] - merged["_patch_date"]).dt.days
    ).astype("float64")
    merged["patch_type"] = merged["_patch_type"].fillna("none")
    merged["patch_id"]   = merged["_patch_id"].fillna("none")
    merged["in_patch_window"] = merged["days_since_patch"].apply(
        lambda d: bool(d <= 30) if pd.notna(d) else False
    )
    merged = merged.drop(columns=["_patch_date", "_patch_type", "_patch_id"], errors="ignore")
    # Restore original row order
    merged = merged.set_index(orig_index if len(orig_index) == len(merged) else merged.index)
    return merged


def run_backtest_for_game(game_id: str, series_df: pd.DataFrame, patches_df: pd.DataFrame,
                          cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs full rolling-origin backtest for a single game.
    Returns (forecasts_df, intervals_df).
    """
    bcfg = cfg["backtest"]
    train_min_days = bcfg["train_min_days"]
    calib_burnin_days = bcfg["calib_burnin_days"]
    horizon = bcfg["horizon"]
    step_days = bcfg["step_days"]
    levels = tuple(float(lvl) for lvl in bcfg["levels"])

    s_game = series_df[series_df["game_id"] == game_id].sort_values("date").reset_index(drop=True)
    p_game = patches_df[patches_df["game_id"] == game_id].sort_values("date").reset_index(drop=True)
    
    unique_dates = s_game["date"].unique()
    n_days = len(unique_dates)
    total_setup_days = train_min_days + calib_burnin_days
    cutoffs = cfg.get("splits", {}).get("cutoffs", {})
    if game_id in cutoffs:
        split_cutoff = pd.to_datetime(cutoffs[game_id])
    else:
        eval_dates = unique_dates[total_setup_days:]
        tune_frac = cfg.get("splits", {}).get("tune_fraction", 0.40)
        tune_idx = max(1, int(len(eval_dates) * tune_frac))
        split_cutoff = pd.to_datetime(eval_dates[min(tune_idx, len(eval_dates) - 1)])

    if n_days < total_setup_days + horizon + 14:
        raise ValueError(
            f"Game {game_id} has {n_days} days, need at least {total_setup_days + horizon + 14} days"
        )

    burnin_start_idx = train_min_days
    burnin_end_idx = total_setup_days
    burnin_end_date = unique_dates[burnin_end_idx]

    # Initialize models
    enabled_models = cfg.get("models", {}).get("list", ["snaive", "lgbm"])
    models = {}
    if "snaive" in enabled_models:
        models["snaive"] = SeasonalNaiveModel(season_length=7)
    if "lgbm" in enabled_models:
        models["lgbm"] = DirectLightGBMForecaster(horizon=horizon)
    if "prophet" in enabled_models:
        try:
            from src.models.prophet_model import ProphetForecaster
            models["prophet"] = ProphetForecaster(horizon=horizon)
        except ImportError:
            print(f"[{game_id}] WARNING: prophet not installed — skipping. Run: pip install prophet")

    # Initialize arms including Arm E (Schedule-Aware)
    icfg = cfg.get("intervals", {})
    arms = {
        "fixed": FixedConformalArm(levels=levels),
        "rolling": RollingConformalArm(
            window_days=icfg.get("rolling_conformal", {}).get("window_days", 90),
            min_samples=icfg.get("rolling_conformal", {}).get("min_samples", 10),
        ),
        "triggered": TriggeredConformalArm(
            k=icfg.get("triggered", {}).get("k", 0.5),
            h=icfg.get("triggered", {}).get("h", 4.0),
            alarm_window_days=icfg.get("triggered", {}).get("alarm_window_days", 21),
            cooldown_days=icfg.get("triggered", {}).get("cooldown_days", 14),
        ),
        "aci": AdaptiveConformalArm(
            gamma=icfg.get("aci", {}).get("gamma", 0.005),
            window_days=icfg.get("aci", {}).get("window_days", 90),  # Bug #4 fix: was reading rolling_conformal key
            min_samples=icfg.get("aci", {}).get("min_samples", 10),
        ),
        "scheduled": ScheduledConformalArm(
            window_days=icfg.get("scheduled", {}).get("window_days", 90),
            patch_window_days=icfg.get("scheduled", {}).get("patch_window_days", 7),
            inflation_factor=icfg.get("scheduled", {}).get("inflation_factor", 1.35),
            min_samples=icfg.get("scheduled", {}).get("min_samples", 10),
        ),
        "hybrid": HybridConformalArm(
            gamma=icfg.get("hybrid", {}).get("gamma", 0.005),
            window_days=icfg.get("hybrid", {}).get("window_days", 90),
            patch_window_days=icfg.get("hybrid", {}).get("patch_window_days", 7),
            inflation_factor=icfg.get("hybrid", {}).get("inflation_factor", 1.35),
            decay_tau=icfg.get("hybrid", {}).get("decay_tau", 2.5),
            min_samples=icfg.get("hybrid", {}).get("min_samples", 10),
        ),
    }

    residual_stores = {m: ResidualStore() for m in models}
    actuals_lookup = dict(zip(s_game["date"], s_game["y_peak"]))

    forecast_records = []
    interval_records = []

    # Delayed feedback queue for adaptive arms (ACI and Hybrid): items pending until target_date <= current_origin
    pending_delayed_feedback: List[dict] = []

    # Step 1: Burn-in roll (train_min_days .. total_setup_days)
    # Collects true out-of-sample residuals for Arm A, CUSUM baseline, and ACI initialization
    print(f"[{game_id}] Phase 1 & 2: Burn-in out-of-sample calibration roll ({burnin_start_idx}..{burnin_end_idx})...")
    for origin_idx in range(burnin_start_idx, burnin_end_idx, step_days):
        origin_date = unique_dates[origin_idx]
        history = s_game[s_game["date"] <= origin_date]

        for m_name, model in models.items():
            if m_name == "snaive":
                preds = model.predict(history, origin_date, horizon)
            else:  # lgbm, prophet — both expose fit_and_predict
                preds = model.fit_and_predict(history, origin_date)

            for _, row in preds.iterrows():
                h = int(row["horizon"])
                td = row["target_date"]
                y_hat = float(row["y_hat"])
                y_true = actuals_lookup.get(td, np.nan)
                residual_stores[m_name].record_forecast(
                    origin_date=origin_date,
                    horizon=h,
                    target_date=td,
                    model=m_name,
                    y_hat=y_hat,
                    y_true=y_true if not np.isnan(y_true) else None,
                )

    # Step 2: Calibrate Arm A and initialize Arm C baseline on burn-in residuals
    print(f"[{game_id}] Calibrating Arm A & CUSUM on burn-in out-of-sample residuals...")
    for m_name in models:
        arms["fixed"].calibrate(residual_stores[m_name], burnin_end_date, m_name, range(1, horizon + 1))
        arms["triggered"].initialize_baseline(residual_stores[m_name], burnin_end_date, m_name)

    # Step 3: Evaluation roll (total_setup_days .. end - horizon)
    print(f"[{game_id}] Phase 3: Evaluation roll ({burnin_end_idx}..{n_days - horizon})...")
    for origin_idx in range(burnin_end_idx, n_days - horizon, step_days):
        origin_date = unique_dates[origin_idx]
        history = s_game[s_game["date"] <= origin_date]

        # 1. Update residual stores with all actuals observable up to origin_date
        observable_actuals = s_game[s_game["date"] <= origin_date]
        for m_name in models:
            residual_stores[m_name].update_actuals(observable_actuals)

        # 2. Process delayed feedback for adaptive arms (ACI and Hybrid):
        # Strictly evaluate past interval forecasts whose target_date <= origin_date (zero lookahead)
        for item in pending_delayed_feedback:
            if not item["processed"] and item["target_date"] <= origin_date:
                y_obs = actuals_lookup.get(item["target_date"], np.nan)
                if not np.isnan(y_obs):
                    was_covered = bool(item["lo"] <= y_obs <= item["hi"])
                    arm_target = item["arm"]
                    if arm_target in arms and hasattr(arms[arm_target], "update_feedback"):
                        arms[arm_target].update_feedback(
                            nominal_level=item["level"],
                            was_covered=was_covered,
                            horizon=item["horizon"],
                        )
                item["processed"] = True

        # Prune processed items to prevent unbounded list growth (Bug #3 fix)
        pending_delayed_feedback = [x for x in pending_delayed_feedback if not x["processed"]]

        # 3. Update CUSUM detector
        for m_name in models:
            arms["triggered"].update_cusum(origin_date, m_name, residual_stores[m_name])

        # 4. Point models fit and predict
        for m_name, model in models.items():
            if m_name == "snaive":
                preds = model.predict(history, origin_date, horizon)
            else:
                preds = model.fit_and_predict(history, origin_date)

            for _, p_row in preds.iterrows():
                h = int(p_row["horizon"])
                td = p_row["target_date"]
                y_hat = float(p_row["y_hat"])
                y_true = actuals_lookup.get(td, np.nan)

                split_label = "TUNE" if pd.to_datetime(origin_date) <= split_cutoff else "EVAL"

                # Store point forecast
                forecast_records.append({
                    "game_id": game_id,
                    "origin_date": origin_date,
                    "horizon": h,
                    "target_date": td,
                    "model": m_name,
                    "y_hat": y_hat,
                    "y_true": float(y_true) if not np.isnan(y_true) else np.nan,
                    "split": split_label,
                })

                # Record in residual store for subsequent origins
                residual_stores[m_name].record_forecast(
                    origin_date=origin_date,
                    horizon=h,
                    target_date=td,
                    model=m_name,
                    y_hat=y_hat,
                    y_true=float(y_true) if not np.isnan(y_true) else None,
                )

                # Generate intervals for all 6 arms and levels
                for arm_name, arm in arms.items():
                    for lvl in levels:
                        lo_base, hi_base = np.nan, np.nan
                        if arm_name == "fixed":
                            lo, hi, width = arm.get_intervals(y_hat, h, lvl)
                        elif arm_name == "rolling":
                            lo, hi, width = arm.get_intervals(y_hat, origin_date, h, lvl, m_name, residual_stores[m_name])
                        elif arm_name == "triggered":
                            lo, hi, width = arm.get_intervals(y_hat, origin_date, h, lvl, m_name, residual_stores[m_name])
                        elif arm_name == "aci":
                            lo, hi, width = arm.get_intervals(y_hat, origin_date, h, lvl, m_name, residual_stores[m_name])
                        elif arm_name == "scheduled":
                            lo, hi, width = arm.get_intervals(y_hat, origin_date, h, lvl, m_name, residual_stores[m_name], p_game)
                        elif arm_name == "hybrid":
                            lo, hi, width, lo_base, hi_base = arm.get_intervals(
                                y_hat, origin_date, h, lvl, m_name, residual_stores[m_name], p_game, return_base=True
                            )

                        covered = bool(lo <= y_true <= hi) if not np.isnan(y_true) else False
                        norm_w = float(width / max(1.0, y_hat))

                        interval_records.append({
                            "game_id": game_id,
                            "origin_date": origin_date,
                            "horizon": h,
                            "target_date": td,
                            "model": m_name,
                            "arm": arm_name,
                            "level": lvl,
                            "lo": lo,
                            "hi": hi,
                            "y_true": float(y_true) if not np.isnan(y_true) else np.nan,
                            "covered": covered,
                            "width": width,
                            "norm_width": norm_w,
                            "split": split_label,
                        })

                        # Queue for delayed feedback (evaluated only when future origin reaches td)
                        # Decouple hybrid: ACI feedback evaluates on uninflated base uncertainty to prevent baseline over-tightening
                        if arm_name in ("aci", "hybrid"):
                            lo_eval = lo_base if arm_name == "hybrid" else lo
                            hi_eval = hi_base if arm_name == "hybrid" else hi
                            pending_delayed_feedback.append({
                                "arm": arm_name,
                                "origin_date": origin_date,
                                "horizon": h,
                                "target_date": td,
                                "level": lvl,
                                "model": m_name,
                                "lo": lo_eval,
                                "hi": hi_eval,
                                "processed": False,
                            })

    f_df = pd.DataFrame(forecast_records)
    i_df = pd.DataFrame(interval_records)

    # Annotate patch context
    i_df = annotate_patch_context(i_df, p_game)

    schemas.validate_forecasts(f_df)
    schemas.validate_intervals(i_df)
    return f_df, i_df
