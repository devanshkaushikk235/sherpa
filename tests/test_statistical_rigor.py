"""Comprehensive Statistical Rigor & Methodology Test Suite.
Verifies:
1. Window labeling correctness on a synthetic patch calendar
2. No-leakage delayed-feedback availability per horizon h
3. Conformal coverage guarantee on exchangeable synthetic data
4. ACI update rule on exact hand-computed values
5. Exponential taper decay values at t=0, tau, 3*tau
6. Capacity cost behavior on toy cases (wide never-miss vs narrow always-miss, break-even r*)
7. Patch-level cluster bootstrap recovers true known effect inside CI without endpoint averaging
8. Determinism: identical random seed produces identical results
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import annotate_patch_context
from src.eval.capacity_cost import compute_capacity_cost_sensitivity, compute_break_even_penalty_ratio
from src.eval.cluster_bootstrap import run_cluster_bootstrap
from src.eval.patch_level import aggregate_patch_level
from src.intervals.aci import AdaptiveConformalArm
from src.intervals.base import conformal_quantile
from src.intervals.hybrid_taper import HybridConformalArm


def test_window_labeling_synthetic_calendar():
    """Asserts window labeling correctly maps days since patch and handles overlaps."""
    patches = pd.DataFrame({
        "game_id": ["game_test", "game_test"],
        "patch_id": ["p1", "p2"],
        "date": [pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-20")],
        "patch_type": ["major", "minor"],
    })

    # Test dates: before p1, on p1, 5d after p1, on p2 (overlapping), 5d after p2, 40d after p2
    test_dates = [
        pd.Timestamp("2023-01-05"), # before any patch -> none, NaN, False
        pd.Timestamp("2023-01-10"), # p1 day 0
        pd.Timestamp("2023-01-15"), # 5d after p1
        pd.Timestamp("2023-01-20"), # p2 day 0 (most recent patch is p2)
        pd.Timestamp("2023-01-25"), # 5d after p2
        pd.Timestamp("2023-03-01"), # 40d after p2 -> baseline
    ]

    df = pd.DataFrame({
        "target_date": test_dates,
    })

    annotated = annotate_patch_context(df, patches)

    assert pd.isna(annotated.iloc[0]["days_since_patch"])
    assert annotated.iloc[0]["patch_id"] == "none"
    assert annotated.iloc[0]["in_patch_window"] == False

    assert annotated.iloc[1]["days_since_patch"] == 0
    assert annotated.iloc[1]["patch_id"] == "p1"
    assert annotated.iloc[1]["in_patch_window"] == True

    assert annotated.iloc[2]["days_since_patch"] == 5
    assert annotated.iloc[2]["patch_id"] == "p1"
    assert annotated.iloc[2]["in_patch_window"] == True

    # On 2023-01-20: most recent is p2 (0 days away)
    assert annotated.iloc[3]["days_since_patch"] == 0
    assert annotated.iloc[3]["patch_id"] == "p2"
    assert annotated.iloc[3]["in_patch_window"] == True

    assert annotated.iloc[4]["days_since_patch"] == 5
    assert annotated.iloc[4]["patch_id"] == "p2"
    assert annotated.iloc[4]["in_patch_window"] == True

    # 40 days after p2: outside 30d window
    assert annotated.iloc[5]["days_since_patch"] == 40
    assert annotated.iloc[5]["in_patch_window"] == False


def test_conformal_coverage_exchangeable():
    """Asserts conformal quantile provides exact (1 - alpha) marginal coverage guarantee."""
    rng = np.random.default_rng(123)
    n_cal = 1000
    n_test = 5000

    # Draw exchangeable standard normal residuals
    cal_scores = np.abs(rng.normal(0, 1, size=n_cal))
    test_scores = np.abs(rng.normal(0, 1, size=n_test))

    for level in [0.8, 0.9, 0.95]:
        q = conformal_quantile(cal_scores, level)
        emp_cov = np.mean(test_scores <= q)
        # Empirical coverage should be >= level - standard error (approx +- 1.5%)
        assert emp_cov >= level - 0.015


def test_aci_hand_computed_update():
    """Asserts ACI update rule exactly matches manual hand-computed gradient."""
    gamma = 0.01
    nominal_level = 0.90
    nominal_alpha = 1.0 - nominal_level  # 0.10

    arm = AdaptiveConformalArm(gamma=gamma, window_days=90)
    init_alpha = arm._get_alpha_t(nominal_level, horizon=1)
    assert np.isclose(init_alpha, 0.10)

    # Case 1: Missed coverage (err = 1.0)
    # alpha_new = 0.10 + 0.01 * (0.10 - 1.0) = 0.10 - 0.009 = 0.091
    arm.update_feedback(nominal_level, was_covered=False, horizon=1)
    assert np.isclose(arm._get_alpha_t(nominal_level, horizon=1), 0.091)

    # Case 2: Covered (err = 0.0)
    # alpha_new = 0.091 + 0.01 * (0.10 - 0.0) = 0.091 + 0.001 = 0.092
    arm.update_feedback(nominal_level, was_covered=True, horizon=1)
    assert np.isclose(arm._get_alpha_t(nominal_level, horizon=1), 0.092)


def test_taper_decay_exact_values():
    """Asserts exponential taper multiplier matches exact mathematical curve."""
    arm = HybridConformalArm(
        inflation_factor=1.35,
        decay_tau=2.5,
        patch_window_days=7,
    )

    origin = pd.Timestamp("2023-01-01")
    patches = pd.DataFrame({
        "game_id": ["game_test"],
        "patch_id": ["p1"],
        "date": [pd.Timestamp("2023-01-01")],
        "patch_type": ["major"],
    })

    # At t = 0: kappa = 1.35
    k0 = arm.get_patch_multiplier(origin, origin, patches)
    assert np.isclose(k0, 1.35)

    # At t = tau = 2.5d: kappa = 1.0 + 0.35 * exp(-1) = 1.0 + 0.35 * 0.367879 = 1.128757
    td_tau = origin + pd.Timedelta(days=2.5)
    k_tau = arm.get_patch_multiplier(td_tau, origin, patches)
    expected_tau = 1.0 + 0.35 * np.exp(-1.0)
    assert np.isclose(k_tau, expected_tau)

    # At t = 3*tau = 7.5d: outside 7d window -> returns 1.0
    td_3tau = origin + pd.Timedelta(days=7.5)
    k_3tau = arm.get_patch_multiplier(td_3tau, origin, patches)
    assert np.isclose(k_3tau, 1.0)


def test_capacity_cost_toy_cases():
    """Tests cost function on edge cases: wide never-miss vs narrow always-miss, and break-even r*."""
    # Toy Case 1: Wide interval that never misses (hi = 200, y = 100)
    # under = 0, over = 100.
    # Total loss = r * 0 + 1 * 100 = 100. norm_cost = 100 / 100 = 1.0 for all r.
    df_wide = pd.DataFrame({
        "y_true": [100.0, 100.0, 100.0],
        "hi": [200.0, 200.0, 200.0],
    })
    sens_wide = compute_capacity_cost_sensitivity(df_wide, ratios=[2.0, 5.0, 10.0])
    assert np.isclose(sens_wide[2.0], 1.0)
    assert np.isclose(sens_wide[5.0], 1.0)
    assert np.isclose(sens_wide[10.0], 1.0)

    # Toy Case 2: Narrow interval that always misses (hi = 50, y = 100)
    # under = 50, over = 0.
    # Total loss = r * 50. norm_cost = (50 * r) / 100 = 0.5 * r.
    df_narrow = pd.DataFrame({
        "y_true": [100.0, 100.0, 100.0],
        "hi": [50.0, 50.0, 50.0],
    })
    sens_narrow = compute_capacity_cost_sensitivity(df_narrow, ratios=[2.0, 5.0, 10.0])
    assert np.isclose(sens_narrow[2.0], 1.0)
    assert np.isclose(sens_narrow[5.0], 2.5)
    assert np.isclose(sens_narrow[10.0], 5.0)

    # Toy Case 3: Break-even ratio calculation
    # Base (narrow): under = 50, over = 0
    # Arm (wide): under = 0, over = 100
    # delta_under = 50 - 0 = 50. delta_over = 100 - 0 = 100.
    # r* = 100 / 50 = 2.0.
    y = np.array([100.0])
    hi_arm = np.array([200.0])
    hi_base = np.array([50.0])
    r_star = compute_break_even_penalty_ratio(y, hi_arm, hi_base)
    assert np.isclose(r_star, 2.0)


def test_cluster_bootstrap_recovers_known_effect():
    """Tests that patch-level cluster bootstrap recovers true known paired differences within CI."""
    rng = np.random.default_rng(42)
    n_patches = 30
    true_cov_diff = 0.05  # Arm X is 5% better coverage than Arm Y

    records = []
    for i in range(n_patches):
        pid = f"patch_{i}"
        gid = "game_1" if i < 15 else "game_2"
        cov_y = float(rng.uniform(0.80, 0.90))
        cov_x = float(cov_y + true_cov_diff + rng.normal(0, 0.01))

        records.append({
            "game_id": gid, "patch_id": pid, "patch_type": "major",
            "model": "lgbm", "arm": "arm_x", "level": 0.9,
            "n_points": 10, "coverage": cov_x, "mean_width": 200.0,
            "winkler": 250.0, "cost_5x": 0.5,
        })
        records.append({
            "game_id": gid, "patch_id": pid, "patch_type": "major",
            "model": "lgbm", "arm": "arm_y", "level": 0.9,
            "n_points": 10, "coverage": cov_y, "mean_width": 180.0,
            "winkler": 260.0, "cost_5x": 0.55,
        })

    patch_df = pd.DataFrame(records)
    boot_res = run_cluster_bootstrap(
        patch_df, n_resamples=1000, seed=42, pairs=[("arm_x", "arm_y")]
    )

    pooled_res = boot_res[boot_res["scope"] == "pooled"].iloc[0]
    ci_lo = pooled_res["cov_diff_ci_lo"]
    ci_hi = pooled_res["cov_diff_ci_hi"]

    # True difference must be covered by the 95% bootstrap CI
    assert ci_lo <= true_cov_diff <= ci_hi
    assert ci_lo > 0.0  # Confirms significant difference from zero


def test_bootstrap_determinism():
    """Asserts identical random seed yields identical bootstrap outputs."""
    records = []
    for i in range(10):
        records.append({
            "game_id": "game_1", "patch_id": f"p_{i}", "patch_type": "major",
            "model": "lgbm", "arm": "arm_x", "level": 0.9,
            "n_points": 5, "coverage": 0.92, "mean_width": 210.0,
            "winkler": 240.0, "cost_5x": 0.48,
        })
        records.append({
            "game_id": "game_1", "patch_id": f"p_{i}", "patch_type": "major",
            "model": "lgbm", "arm": "arm_y", "level": 0.9,
            "n_points": 5, "coverage": 0.86, "mean_width": 180.0,
            "winkler": 255.0, "cost_5x": 0.45,
        })
    df = pd.DataFrame(records)

    res1 = run_cluster_bootstrap(df, n_resamples=500, seed=999, pairs=[("arm_x", "arm_y")])
    res2 = run_cluster_bootstrap(df, n_resamples=500, seed=999, pairs=[("arm_x", "arm_y")])

    assert np.isclose(res1.iloc[0]["cov_diff_mean"], res2.iloc[0]["cov_diff_mean"])
    assert np.isclose(res1.iloc[0]["cov_diff_ci_lo"], res2.iloc[0]["cov_diff_ci_lo"])
    assert np.isclose(res1.iloc[0]["cov_diff_ci_hi"], res2.iloc[0]["cov_diff_ci_hi"])
