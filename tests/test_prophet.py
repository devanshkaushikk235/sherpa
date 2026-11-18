"""Unit tests for Prophet forecasting baseline."""
import pandas as pd
import pytest

from src.models.prophet_model import ProphetForecaster, PROPHET_AVAILABLE


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="prophet not installed")
def test_prophet_forecaster_fit_and_predict():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    # Synthetic series with weekly pattern and positive values
    values = [1000 + 200 * (i % 7) + i * 5 for i in range(60)]
    history = pd.DataFrame({"date": dates, "y_peak": values})
    origin = dates[-1]

    model = ProphetForecaster(
        horizon=7,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    try:
        preds = model.fit_and_predict(history, origin)
    except RuntimeError as e:
        if "WinError 623" in str(e) or "DLL Relocation" in str(e):
            pytest.skip(f"Windows Python 3.14 CmdStan DLL relocation issue: {e}")
        raise

    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == 7
    assert set(preds.columns) == {"horizon", "target_date", "y_hat"}
    assert (preds["horizon"] == list(range(1, 8))).all()
    assert (preds["y_hat"] > 0).all()
    assert not preds["y_hat"].isna().any()


@pytest.mark.skipif(not PROPHET_AVAILABLE, reason="prophet not installed")
def test_prophet_forecaster_fallback_on_short_history():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    history = pd.DataFrame({"date": dates, "y_peak": [500] * 10})
    origin = dates[-1]

    model = ProphetForecaster(horizon=3)
    try:
        preds = model.fit_and_predict(history, origin)
    except RuntimeError as e:
        if "WinError 623" in str(e) or "DLL Relocation" in str(e):
            pytest.skip(f"Windows Python 3.14 CmdStan DLL relocation issue: {e}")
        raise

    assert len(preds) == 3
    assert (preds["y_hat"] > 0).all()
