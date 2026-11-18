"""Unit tests for conformal prediction intervals, bounds, and calibration."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.residual_store import ResidualStore
from src.intervals.aci import AdaptiveConformalArm
from src.intervals.base import conformal_quantile, log_bounds_to_raw
from src.intervals.fixed import FixedConformalArm
from src.intervals.rolling import RollingConformalArm
from src.intervals.scheduled import ScheduledConformalArm


def test_conformal_quantile_bounds():
    """Check finite-sample quantile calculation properties."""
    scores = np.linspace(0.1, 1.0, 100)
    q80 = conformal_quantile(scores, 0.8)
    q90 = conformal_quantile(scores, 0.9)
    assert q80 <= q90
    assert 0.75 <= q80 <= 0.85
    assert 0.85 <= q90 <= 0.95


def test_log_bounds_non_negative():
    """Assert lower bounds are always non-negative in raw scale."""
    y_hat = 10.0
    q_large = 5.0  # huge log quantile
    lo, hi, width = log_bounds_to_raw(y_hat, q_large)
    assert lo >= 0.0
    assert hi > y_hat
    assert width == hi - lo


def test_fixed_conformal_requires_calibration():
    """Assert FixedConformalArm raises error if used before calibrate()."""
    arm = FixedConformalArm(levels=(0.8, 0.9))
    with pytest.raises(RuntimeError):
        arm.get_intervals(100.0, horizon=1, level=0.8)


def test_rolling_conformal_coverage_on_stationary():
    """Assert empirical coverage on stationary residuals hits near nominal."""
    np.random.seed(42)
    rs = ResidualStore()
    origin = pd.Timestamp("2023-04-01")
    
    # Generate 100 stationary past predictions with log-normal noise
    true_sigma = 0.15
    for i in range(100):
        td = origin - pd.Timedelta(days=100 - i)
        y_true = float(np.random.uniform(500, 1000))
        # log residual is normal
        noise = float(np.random.normal(0, true_sigma))
        y_hat = float(np.expm1(np.log1p(y_true) + noise))
        rs.record_forecast(
            origin_date=td - pd.Timedelta(days=1),
            horizon=1,
            target_date=td,
            model="snaive",
            y_hat=y_hat,
            y_true=y_true,
        )

    arm = RollingConformalArm(window_days=90)
    
    # Test on 200 out-of-sample test draws
    covered_80 = []
    covered_90 = []
    for _ in range(200):
        test_y = float(np.random.uniform(500, 1000))
        noise = float(np.random.normal(0, true_sigma))
        test_hat = float(np.expm1(np.log1p(test_y) + noise))
        
        lo80, hi80, _ = arm.get_intervals(test_hat, origin, horizon=1, level=0.8, model="snaive", residual_store=rs)
        lo90, hi90, _ = arm.get_intervals(test_hat, origin, horizon=1, level=0.9, model="snaive", residual_store=rs)
        
        covered_80.append(lo80 <= test_y <= hi80)
        covered_90.append(lo90 <= test_y <= hi90)

    emp_cov_80 = np.mean(covered_80)
    emp_cov_90 = np.mean(covered_90)

    # Conformal guarantee: should be close to nominal
    assert 0.72 <= emp_cov_80 <= 0.88, f"Expected ~0.80, got {emp_cov_80}"
    assert 0.83 <= emp_cov_90 <= 0.96, f"Expected ~0.90, got {emp_cov_90}"


def test_scheduled_conformal_inflates_on_patch():
    """Assert Arm E inflates interval width when target date falls in patch window."""
    rs = ResidualStore()
    origin = pd.Timestamp("2023-05-01")
    for i in range(30):
        td = origin - pd.Timedelta(days=30 - i)
        rs.record_forecast(
            origin_date=td - pd.Timedelta(days=1),
            horizon=1,
            target_date=td,
            model="snaive",
            y_hat=100.0,
            y_true=105.0,
        )

    # Patch occurs on 2023-05-03 (target date for horizon 2)
    patches = pd.DataFrame({
        "game_id": ["game_test"],
        "date": [pd.Timestamp("2023-05-03")],
        "patch_id": ["p1"],
        "patch_type": ["major"],
    })

    arm = ScheduledConformalArm(window_days=90, patch_window_days=7, inflation_factor=1.5, min_samples=5)
    
    # Horizon 1 (2023-05-02): before patch
    lo_normal, hi_normal, width_normal = arm.get_intervals(
        y_hat=100.0, origin=origin, horizon=1, level=0.9, model="snaive", residual_store=rs, patches_df=patches
    )
    # Horizon 2 (2023-05-03): on patch date
    lo_patch, hi_patch, width_patch = arm.get_intervals(
        y_hat=100.0, origin=origin, horizon=2, level=0.9, model="snaive", residual_store=rs, patches_df=patches
    )

    assert width_patch > width_normal
    assert lo_patch < lo_normal
    assert hi_patch > hi_normal


def test_aci_delayed_feedback_horizon_tracking():
    """Assert AdaptiveConformalArm tracks feedback per horizon independently."""
    rs = ResidualStore()
    origin = pd.Timestamp("2023-05-01")
    for i in range(30):
        td = origin - pd.Timedelta(days=30 - i)
        rs.record_forecast(
            origin_date=td - pd.Timedelta(days=1),
            horizon=1,
            target_date=td,
            model="snaive",
            y_hat=100.0,
            y_true=105.0,
        )

    arm = AdaptiveConformalArm(gamma=0.05, window_days=90, min_samples=5)
    
    # Initial state
    alpha_h1_init = arm._get_alpha_t(nominal_level=0.9, horizon=1)
    alpha_h7_init = arm._get_alpha_t(nominal_level=0.9, horizon=7)
    assert np.isclose(alpha_h1_init, 0.10)
    assert np.isclose(alpha_h7_init, 0.10)

    # Receive repeated missed coverage errors on horizon 1 ONLY
    for _ in range(5):
        arm.update_feedback(nominal_level=0.9, was_covered=False, horizon=1)

    alpha_h1_after = arm._get_alpha_t(nominal_level=0.9, horizon=1)
    alpha_h7_after = arm._get_alpha_t(nominal_level=0.9, horizon=7)

    # When coverage is missed, noncoverage rate alpha decreases (effective coverage 1 - alpha increases to widen intervals)
    assert alpha_h1_after < alpha_h1_init
    # Horizon 7 without errors retains nominal alpha
    assert alpha_h7_after > alpha_h1_after
    # Effective coverage level for horizon 1 is higher than horizon 7
    assert (1.0 - alpha_h1_after) > (1.0 - alpha_h7_after)


def test_hybrid_taper_decay_and_feedback():
    """Assert HybridConformalArm exhibits exponential decay and ACI adaptation."""
    from src.intervals.hybrid_taper import HybridConformalArm

    arm = HybridConformalArm(
        gamma=0.05,
        window_days=90,
        patch_window_days=7,
        inflation_factor=1.35,
        decay_tau=2.5,
        min_samples=5,
    )

    origin = pd.Timestamp("2023-05-01")
    patches = pd.DataFrame({
        "game_id": ["game_test"],
        "date": [pd.Timestamp("2023-05-01")],
        "patch_id": ["p1"],
        "patch_type": ["major"],
    })

    # Day 0 (patch day): multiplier = 1.35
    m0 = arm.get_patch_multiplier(origin, origin, patches)
    assert np.isclose(m0, 1.35)

    # Day 3: multiplier decays exponentially
    td3 = origin + pd.Timedelta(days=3)
    m3 = arm.get_patch_multiplier(td3, origin, patches)
    expected_m3 = 1.0 + 0.35 * np.exp(-3.0 / 2.5)
    assert np.isclose(m3, expected_m3)
    assert m3 < m0 and m3 > 1.0

    # Day 10 (outside 7-day window): multiplier is 1.0
    td10 = origin + pd.Timedelta(days=10)
    m10 = arm.get_patch_multiplier(td10, origin, patches)
    assert np.isclose(m10, 1.0)

    # ACI feedback adaptation: missed coverage decreases alpha
    arm.update_feedback(nominal_level=0.9, was_covered=False, horizon=1)
    assert arm._get_alpha_t(0.9, horizon=1) < 0.10

def test_scheduled_inflates_only_patch_window():
    """Assert Arm E inflates ONLY target days within patch_window_days of a patch.

    Previously a bug caused ALL days in a forecast horizon to be inflated whenever
    any patch occurred somewhere in that horizon.  This test pins the correct behaviour:
    - h=2 target (day after patch) → inflated (within 7-day window)
    - h=14 target (12 days AFTER patch day) → inflated (still within 7-day window if ≤7)
    - h=20 target (18 days after patch) → NOT inflated (outside window)
    """
    rs = ResidualStore()
    origin = pd.Timestamp("2023-05-01")
    for i in range(30):
        td = origin - pd.Timedelta(days=30 - i)
        rs.record_forecast(
            origin_date=td - pd.Timedelta(days=1),
            horizon=1,
            target_date=td,
            model="snaive",
            y_hat=100.0,
            y_true=105.0,
        )

    # Patch on day 2 of forecast (2023-05-03)
    patch_date = pd.Timestamp("2023-05-03")
    patches = pd.DataFrame({
        "game_id":    ["game_test"],
        "date":       [patch_date],
        "patch_id":   ["p1"],
        "patch_type": ["major"],
    })

    arm = ScheduledConformalArm(window_days=90, patch_window_days=7,
                                inflation_factor=2.0, min_samples=5)

    # h=2: target = 2023-05-03 (patch day itself) → inflated
    lo_patch, hi_patch, w_patch = arm.get_intervals(
        100.0, origin, horizon=2, level=0.9,
        model="snaive", residual_store=rs, patches_df=patches,
    )
    # h=9: target = 2023-05-10, 7 days after patch → boundary, still inflated
    lo_7d, hi_7d, w_7d = arm.get_intervals(
        100.0, origin, horizon=9, level=0.9,
        model="snaive", residual_store=rs, patches_df=patches,
    )
    # h=11: target = 2023-05-12, 9 days after patch → NOT inflated
    lo_out, hi_out, w_out = arm.get_intervals(
        100.0, origin, horizon=11, level=0.9,
        model="snaive", residual_store=rs, patches_df=patches,
    )
    # h=1: target = 2023-05-02, BEFORE the patch → NOT inflated
    lo_pre, hi_pre, w_pre = arm.get_intervals(
        100.0, origin, horizon=1, level=0.9,
        model="snaive", residual_store=rs, patches_df=patches,
    )

    assert w_patch > w_out,  "Patch-day horizon must be wider than outside-window horizon"
    assert w_7d >= w_out,    "Boundary day (day 7) must be >= outside-window width"
    assert np.isclose(w_out, w_pre, rtol=1e-6), \
        "Days outside window and days before patch must have equal (non-inflated) width"
