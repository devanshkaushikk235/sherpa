"""Seasonal-Naive point forecasting baseline.
For horizon h in 1..14, predicts y_{t+h} using the most recent observation matching that day of week:
(t + h) - 7 * ceil(h / 7) <= t.
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class SeasonalNaiveModel:
    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def predict(self, history: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """
        history: DataFrame with columns ['date', 'y_peak'] sorted by date, where date <= origin.
        origin: Timestamp representing the origin date t.
        horizon: number of days to forecast forward (1..horizon).
        returns: DataFrame with columns ['horizon', 'target_date', 'y_hat']
        """
        # Ensure history is strictly <= origin
        h_df = history[history["date"] <= origin].copy()
        if h_df.empty:
            raise ValueError(f"No history available on or before origin {origin}")
        
        # Build lookup table date -> y_peak
        lookup = dict(zip(h_df["date"], h_df["y_peak"]))
        # Also keep last valid value as emergency fallback
        valid_series = h_df["y_peak"].dropna()
        last_valid = valid_series.iloc[-1] if not valid_series.empty else 1.0

        records = []
        for h in range(1, horizon + 1):
            target_date = origin + pd.Timedelta(days=h)
            # Find the most recent date with the same day-of-week that is <= origin
            # Lag is 7 * ceil(h / 7) days prior to target_date
            k = int(np.ceil(h / self.season_length))
            ref_date = target_date - pd.Timedelta(days=k * self.season_length)
            y_hat = lookup.get(ref_date, np.nan)
            if np.isnan(y_hat):
                # Fallback to the most recent known observation <= origin
                y_hat = last_valid
            records.append({
                "horizon": h,
                "target_date": target_date,
                "y_hat": float(max(1.0, y_hat)),
            })
        return pd.DataFrame(records)
