"""Arm B: Rolling Conformal Intervals.
Calibrated at each origin on a trailing window (default 90 days) of observable residuals.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class RollingConformalArm:
    def __init__(self, window_days: int = 90, min_samples: int = 10):
        self.window_days = window_days
        self.min_samples = min_samples

    def get_intervals(self, y_hat: float, origin: pd.Timestamp, horizon: int,
                      level: float, model: str, residual_store) -> Tuple[float, float, float]:
        """
        Retrieves observable residuals with target_date <= origin within the last window_days.
        """
        scores = residual_store.get_observable_residuals(
            current_origin=origin,
            model=model,
            horizon=horizon,
            max_window_days=self.window_days,
        )
        if len(scores) < self.min_samples:
            # Pool across horizons if samples are scarce
            scores = residual_store.get_observable_residuals(
                current_origin=origin,
                model=model,
                max_window_days=self.window_days,
            )

        q = conformal_quantile(scores, level)
        return log_bounds_to_raw(y_hat, q)
