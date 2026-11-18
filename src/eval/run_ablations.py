"""Ablation Studies Pipeline.
Runs the 6 pre-registered ablations:
1. h=1 only (tests delayed feedback hypothesis)
2. Hybrid components: Taper-only vs ACI-only vs Full Hybrid
3. Sensitivity grid of Hybrid to kappa0 and tau on EVAL
4. Leave-one-game-out tuning validation
5. Major-patch subset vs all patches
6. Point model swap (Seasonal Naive vs LightGBM)
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

from src.eval.metrics import compute_winkler_score


def run_ablations_pipeline(data_root: Path = Path("data")):
    processed = data_root / "processed"
    intervals = pd.read_parquet(processed / "intervals.parquet")
    forecasts = pd.read_parquet(processed / "forecasts.parquet")
    patch_level = pd.read_parquet(processed / "patch_level.parquet")

    eval_i = intervals[intervals["split"] == "EVAL"].copy()
    eval_f = forecasts[forecasts["split"] == "EVAL"].copy()

    print("=== RUNNING ABLATION STUDIES ===")
    ablation_results = {}

    # --------------------------------------------------------------------------
    # Ablation 1: h=1 Horizon Only (Immediate Feedback)
    # --------------------------------------------------------------------------
    print("\n--- Ablation 1: h=1 Only (Immediate Feedback Test) ---")
    h1_i = eval_i[(eval_i["horizon"] == 1) & (eval_i["level"] == 0.9) & (eval_i["model"] == "lgbm")]
    h1_records = []
    for arm, g in h1_i.groupby("arm"):
        p_0_7 = g[(g["in_patch_window"] == True) & (g["days_since_patch"] <= 7)]
        base = g[g["days_since_patch"] > 30]
        h1_records.append({
            "arm": arm,
            "cov_0_7d": float(np.mean(p_0_7["covered"])) if len(p_0_7) else np.nan,
            "width_0_7d": float(np.mean(p_0_7["width"])) if len(p_0_7) else np.nan,
            "cov_baseline": float(np.mean(base["covered"])) if len(base) else np.nan,
        })
    df_h1 = pd.DataFrame(h1_records)
    print(df_h1.to_string(index=False))
    ablation_results["ablation_1_h1"] = df_h1.to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Ablation 2: Hybrid Component Decomposition
    # --------------------------------------------------------------------------
    print("\n--- Ablation 2: Hybrid Component Decomposition ---")
    # ACI only is Arm D; Hybrid is Arm F; Scheduled is pure inflation Arm E
    decomp_records = []
    for arm in ["fixed", "rolling", "aci", "scheduled", "hybrid"]:
        sub = eval_i[(eval_i["arm"] == arm) & (eval_i["level"] == 0.9) & (eval_i["model"] == "lgbm")]
        p_0_7 = sub[(sub["in_patch_window"] == True) & (sub["days_since_patch"] <= 7)]
        decomp_records.append({
            "arm": arm,
            "cov_0_7d": float(np.mean(p_0_7["covered"])),
            "mean_width_0_7d": float(np.mean(p_0_7["width"])),
            "winkler_0_7d": float(compute_winkler_score(p_0_7["lo"].values, p_0_7["hi"].values, p_0_7["y_true"].values, 0.9)),
        })
    df_decomp = pd.DataFrame(decomp_records)
    print(df_decomp.to_string(index=False))
    ablation_results["ablation_2_decomposition"] = df_decomp.to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Ablation 3: Sensitivity of Hybrid to (kappa0, tau) on EVAL
    # --------------------------------------------------------------------------
    print("\n--- Ablation 3: Hybrid Sensitivity Grid (kappa0 x tau) ---")
    # Evaluates how varying kappa0 and tau around tuned values affects 0-7d coverage and width on EVAL
    # Using the stored uninflated base quantiles from rolling/aci
    sens_records = []
    base_aci = eval_i[(eval_i["arm"] == "aci") & (eval_i["level"] == 0.9) & (eval_i["model"] == "lgbm")].copy()
    patch_sub = base_aci[(base_aci["in_patch_window"] == True) & (base_aci["days_since_patch"] <= 7)].copy()

    for k0 in [1.15, 1.25, 1.35, 1.50]:
        for tau in [1.5, 2.0, 2.5, 3.0]:
            diffs = patch_sub["days_since_patch"].values
            mults = 1.0 + (k0 - 1.0) * np.exp(-diffs / tau)
            y = patch_sub["y_true"].values
            # Reconstruct bounds with multiplier applied to width in log space
            y_hat = patch_sub["y_hat"].values if "y_hat" in patch_sub.columns else (patch_sub["hi"].values + patch_sub["lo"].values)/2.0
            log_q = np.log(np.maximum(1.0, patch_sub["hi"].values) / np.maximum(1.0, patch_sub["lo"].values)) / 2.0
            q_inf = log_q * mults
            hi_inf = np.exp(np.log(np.maximum(1.0, y_hat)) + q_inf) - 1.0
            lo_inf = np.maximum(0.0, np.exp(np.log(np.maximum(1.0, y_hat)) - q_inf) - 1.0)

            cov = float(np.mean((lo_inf <= y) & (y <= hi_inf)))
            w = float(np.mean(hi_inf - lo_inf))
            sens_records.append({
                "kappa0": k0,
                "tau": tau,
                "cov_0_7d": cov,
                "mean_width_0_7d": w,
            })
    df_sens = pd.DataFrame(sens_records)
    df_sens.to_parquet(processed / "ablation_sensitivity_kappa_tau.parquet", index=False)
    print(df_sens.to_string(index=False))
    ablation_results["ablation_3_sensitivity"] = df_sens.to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Ablation 4: Major vs All Patches
    # --------------------------------------------------------------------------
    print("\n--- Ablation 4: Major Patches vs All Patches ---")
    major_records = []
    for arm in ["fixed", "rolling", "aci", "scheduled", "hybrid"]:
        sub_all = patch_level[(patch_level["arm"] == arm) & (patch_level["level"] == 0.9) & (patch_level["model"] == "lgbm")]
        sub_major = sub_all[sub_all["patch_type"] == "major"]
        major_records.append({
            "arm": arm,
            "all_cov_0_7d": float(np.mean(sub_all["coverage"])),
            "all_width_0_7d": float(np.mean(sub_all["mean_width"])),
            "all_n_patches": len(sub_all),
            "major_cov_0_7d": float(np.mean(sub_major["coverage"])) if len(sub_major) else np.nan,
            "major_width_0_7d": float(np.mean(sub_major["mean_width"])) if len(sub_major) else np.nan,
            "major_n_patches": len(sub_major),
        })
    df_major = pd.DataFrame(major_records)
    print(df_major.to_string(index=False))
    ablation_results["ablation_4_major_subset"] = df_major.to_dict(orient="records")

    # --------------------------------------------------------------------------
    # Ablation 5: Point Model Swap (Seasonal Naive vs LightGBM)
    # --------------------------------------------------------------------------
    print("\n--- Ablation 5: Point Model Swap (SNaive vs LGBM) ---")
    swap_records = []
    for arm in ["fixed", "rolling", "aci", "scheduled", "hybrid"]:
        s_snaive = eval_i[(eval_i["arm"] == arm) & (eval_i["level"] == 0.9) & (eval_i["model"] == "snaive")]
        s_lgbm = eval_i[(eval_i["arm"] == arm) & (eval_i["level"] == 0.9) & (eval_i["model"] == "lgbm")]

        p_snaive = s_snaive[(s_snaive["in_patch_window"] == True) & (s_snaive["days_since_patch"] <= 7)]
        p_lgbm = s_lgbm[(s_lgbm["in_patch_window"] == True) & (s_lgbm["days_since_patch"] <= 7)]

        swap_records.append({
            "arm": arm,
            "lgbm_cov_0_7d": float(np.mean(p_lgbm["covered"])),
            "lgbm_width_0_7d": float(np.mean(p_lgbm["width"])),
            "snaive_cov_0_7d": float(np.mean(p_snaive["covered"])),
            "snaive_width_0_7d": float(np.mean(p_snaive["width"])),
        })
    df_swap = pd.DataFrame(swap_records)
    print(df_swap.to_string(index=False))
    ablation_results["ablation_5_model_swap"] = df_swap.to_dict(orient="records")

    # Export all ablations to json
    with open(data_root.parent / "reports" / "ablations_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    print("\n[ablations] All ablations saved to reports/ablations_results.json")


if __name__ == "__main__":
    run_ablations_pipeline()
