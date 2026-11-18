"""Unit tests enforcing strict zero-leakage invariants across backtest components."""
import numpy as np
import pandas as pd
import pytest

from src.backtest.residual_store import ResidualStore
from src.intervals.triggered import TriggeredConformalArm
from src.models.lgbm_direct import extract_features_at_origin
from src.models.snaive import SeasonalNaiveModel


def test_residual_store_zero_lookahead():
    """Verify that get_observable_residuals never returns targets after current_origin."""
    rs = ResidualStore()
    origin = pd.Timestamp("2023-01-10")

    # Target in the past (observable)
    rs.record_forecast(
        origin_date=pd.Timestamp("2023-01-01"),
        horizon=5,
        target_date=pd.Timestamp("2023-01-06"),
        model="lgbm",
        y_hat=100.0,
        y_true=105.0,
    )
    # Target in the future relative to origin (not yet observable)
    rs.record_forecast(
        origin_date=pd.Timestamp("2023-01-08"),
        horizon=5,
        target_date=pd.Timestamp("2023-01-13"),
        model="lgbm",
        y_hat=120.0,
        y_true=130.0,
    )

    scores = rs.get_observable_residuals(current_origin=origin, model="lgbm")
    assert len(scores) == 1
    # Only the 2023-01-06 residual should be included
    expected_score = abs(np.log1p(105.0) - np.log1p(100.0))
    assert np.isclose(scores[0], expected_score, atol=1e-5)


def test_residual_store_no_future_residuals():
    """Assert that a record with target_date == current_origin is NOT included.

    The invariant is strict: target_date <= current_origin means the value has
    been observed.  When target_date equals origin (same day), the check
    must still pass (it is observable); but target_date one day ahead must not.
    """
    rs = ResidualStore()
    origin = pd.Timestamp("2023-06-01")

    # Record whose target is exactly one day AHEAD of origin (not observable)
    rs.record_forecast(
        origin_date=pd.Timestamp("2023-05-18"),
        horizon=14,
        target_date=pd.Timestamp("2023-06-02"),   # strictly after origin
        model="lgbm",
        y_hat=500.0,
        y_true=520.0,
    )

    scores = rs.get_observable_residuals(current_origin=origin, model="lgbm")
    assert len(scores) == 0, (
        f"Expected 0 observable residuals at origin {origin}, got {len(scores)}"
    )

    # But a record with target == origin IS observable
    rs.record_forecast(
        origin_date=pd.Timestamp("2023-05-18"),
        horizon=14,
        target_date=origin,   # exactly on origin
        model="lgbm",
        y_hat=500.0,
        y_true=505.0,
    )
    scores2 = rs.get_observable_residuals(current_origin=origin, model="lgbm")
    assert len(scores2) == 1, "target_date == current_origin should be observable"


def test_snaive_no_future_access():
    """Verify SeasonalNaive only queries dates <= origin."""
    dates = pd.date_range("2023-01-01", "2023-01-30", freq="D")
    df = pd.DataFrame({"date": dates, "y_peak": np.arange(100, 100 + len(dates), dtype=float)})

    origin = pd.Timestamp("2023-01-15")
    model = SeasonalNaiveModel(season_length=7)
    # Even if future dates are present in df, predict should only use <= origin
    preds = model.predict(df, origin, horizon=14)
    assert len(preds) == 14
    assert (preds["target_date"] > origin).all()
    # Check that h=7 prediction matches origin (2023-01-15 value)
    val_origin = df.loc[df["date"] == origin, "y_peak"].values[0]
    pred_h7 = preds.loc[preds["horizon"] == 7, "y_hat"].values[0]
    assert np.isclose(pred_h7, val_origin)


def test_lgbm_features_no_future_leakage():
    """Verify feature extractor at origin t ignores data after t."""
    dates = pd.date_range("2023-01-01", "2023-03-01", freq="D")
    df = pd.DataFrame({"date": dates, "y_peak": np.random.uniform(500, 1000, size=len(dates))})

    origin = pd.Timestamp("2023-02-15")
    f_full = extract_features_at_origin(df, origin)

    # Truncate df to origin
    df_truncated = df[df["date"] <= origin].copy()
    f_trunc = extract_features_at_origin(df_truncated, origin)

    assert f_full is not None
    assert f_trunc is not None
    for k in f_full:
        assert np.isclose(f_full[k], f_trunc[k])

