"""Prophet forecasting baseline.

Wraps Facebook/Meta's Prophet for direct multi-horizon forecasting.
Trained once per origin on all available history, predicts h=1..horizon days ahead.
Used as a named industry-standard baseline alongside LightGBM.
"""
from typing import Optional
import numpy as np
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecaster:
    """
    Multi-horizon forecaster using Facebook Prophet.

    Fits a Prophet model on history up to `origin`, then reads off
    predictions at horizons 1..horizon days ahead.
    """

    def __init__(
        self,
        horizon: int = 14,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        random_state: int = 42,
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "prophet is not installed. Run: pip install prophet"
            )
        self.horizon = horizon
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.random_state = random_state
        self._model: Optional[Prophet] = None

    def fit_and_predict(
        self, history: pd.DataFrame, origin: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Fits Prophet on history up to and including `origin`, predicts
        horizons 1..self.horizon days ahead.

        Args:
            history: DataFrame with columns ['date', 'y_peak'], date <= origin.
            origin: Forecast origin date.

        Returns:
            DataFrame with columns ['horizon', 'target_date', 'y_hat'].
        """
        df = (
            history[history["date"] <= origin]
            .sort_values("date")
            .copy()
        )
        # Prophet requires columns 'ds' and 'y'; log1p for stability
        df = df.dropna(subset=["y_peak"])
        df = df[df["y_peak"] > 0].copy()
        if len(df) < 30:
            # Fall back to seasonal naive if not enough history
            return self._snaive_fallback(df, origin)

        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df["date"]),
            "y": np.log1p(df["y_peak"].values),
        })

        model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
        )
        # Suppress Stan output
        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        model.fit(prophet_df)
        self._model = model

        # Build future dataframe for horizons 1..horizon
        future_dates = pd.DataFrame({
            "ds": [
                pd.Timestamp(origin) + pd.Timedelta(days=h)
                for h in range(1, self.horizon + 1)
            ]
        })
        forecast = model.predict(future_dates)

        results = []
        for h in range(1, self.horizon + 1):
            row = forecast.iloc[h - 1]
            y_hat_log = float(row["yhat"])
            y_hat = float(max(1.0, np.expm1(y_hat_log)))
            results.append({
                "horizon": h,
                "target_date": pd.Timestamp(origin) + pd.Timedelta(days=h),
                "y_hat": y_hat,
            })

        return pd.DataFrame(results)

    def _snaive_fallback(
        self, df: pd.DataFrame, origin: pd.Timestamp
    ) -> pd.DataFrame:
        """Seasonal naive fallback when history is too short for Prophet."""
        lookup = dict(zip(df["date"], df["y_peak"]))
        last_valid = float(df["y_peak"].dropna().iloc[-1]) if not df.empty else 1.0
        records = []
        for h in range(1, self.horizon + 1):
            target_date = pd.Timestamp(origin) + pd.Timedelta(days=h)
            k = int(np.ceil(h / 7))
            ref_date = target_date - pd.Timedelta(days=k * 7)
            y_hat = float(lookup.get(ref_date, last_valid))
            records.append({
                "horizon": h,
                "target_date": target_date,
                "y_hat": max(1.0, y_hat),
            })
        return pd.DataFrame(records)
