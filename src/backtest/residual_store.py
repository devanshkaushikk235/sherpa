"""Residual Store for Conformal Recalibration.
Stores historical forecasts and computes log-scale nonconformity scores:
e_{t, h} = |log(y_{t+h} + 1) - log(y_hat_{t+h} + 1)|.
Strictly returns residuals whose target_date <= current_origin (zero lookahead).

Performance note: records are appended to a list and a DataFrame cache is rebuilt lazily
on the first call after any mutation.  This reduces get_observable_residuals from O(N)
per call to O(1) amortized for the cache rebuild plus vectorised boolean masking.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ResidualStore:
    def __init__(self):
        self._records: List[dict] = []
        self._cache: Optional[pd.DataFrame] = None   # lazy, rebuilt on demand
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invalidate(self) -> None:
        self._dirty = True

    def _get_df(self) -> pd.DataFrame:
        """Return the cached DataFrame, rebuilding it if the store is dirty."""
        if self._dirty or self._cache is None:
            if self._records:
                self._cache = pd.DataFrame(self._records)
                # Ensure datetime dtype for fast comparisons
                self._cache["target_date"] = pd.to_datetime(self._cache["target_date"])
                self._cache["origin_date"] = pd.to_datetime(self._cache["origin_date"])
            else:
                self._cache = pd.DataFrame(columns=[
                    "origin_date", "horizon", "target_date",
                    "model", "y_hat", "y_true", "log_residual",
                ])
            self._dirty = False
        return self._cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_forecast(self, origin_date: pd.Timestamp, horizon: int,
                        target_date: pd.Timestamp, model: str,
                        y_hat: float, y_true: Optional[float] = None) -> None:
        log_res = None
        if y_true is not None and not np.isnan(y_true) and y_true > 0 and y_hat > 0:
            log_res = float(abs(np.log1p(y_true) - np.log1p(y_hat)))

        self._records.append({
            "origin_date": origin_date,
            "horizon": horizon,
            "target_date": target_date,
            "model": model,
            "y_hat": y_hat,
            "y_true": y_true,
            "log_residual": log_res,
        })
        self._invalidate()

    def update_actuals(self, actuals_series: pd.DataFrame) -> None:
        """
        Updates y_true and log_residual for any records matching date in actuals_series.
        actuals_series must have ['date', 'y_peak'].
        """
        lookup = dict(zip(actuals_series["date"], actuals_series["y_peak"]))
        updated = False
        for rec in self._records:
            if rec["y_true"] is None or (
                isinstance(rec["y_true"], float) and np.isnan(rec["y_true"])
            ):
                td = rec["target_date"]
                if td in lookup:
                    val = lookup[td]
                    if not np.isnan(val) and val > 0:
                        rec["y_true"] = float(val)
                        if rec["y_hat"] > 0:
                            rec["log_residual"] = float(
                                abs(np.log1p(val) - np.log1p(rec["y_hat"]))
                            )
                        updated = True
        if updated:
            self._invalidate()

    def get_observable_residuals(self, current_origin: pd.Timestamp, model: str,
                                 horizon: Optional[int] = None,
                                 max_window_days: Optional[int] = None,
                                 min_target_date: Optional[pd.Timestamp] = None,
                                 max_target_date: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Returns an array of log-residuals strictly observable at current_origin
        (target_date <= current_origin).  Vectorised via DataFrame masking.
        """
        df = self._get_df()
        if df.empty:
            return np.array([], dtype=np.float64)

        current_origin = pd.to_datetime(current_origin)
        mask = (
            (df["model"] == model)
            & (df["target_date"] <= current_origin)
            & df["log_residual"].notna()
        )
        if horizon is not None:
            mask &= df["horizon"] == horizon
        if max_window_days is not None:
            mask &= (current_origin - df["target_date"]).dt.days <= max_window_days
        if min_target_date is not None:
            mask &= df["target_date"] >= pd.to_datetime(min_target_date)
        if max_target_date is not None:
            mask &= df["target_date"] <= pd.to_datetime(max_target_date)

        return df.loc[mask, "log_residual"].to_numpy(dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        return self._get_df().copy()
