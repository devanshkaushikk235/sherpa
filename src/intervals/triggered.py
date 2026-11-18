"""Arm C: Change-Triggered Conformal Intervals.
Monitors observable standardized residuals via CUSUM.
Baseline mu_0, sigma_0 are computed STRICTLY on out-of-sample burn-in residuals (zero future leakage).
On drift alarm, contracts the calibration window to alarm_window_days for fast adaptation.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class TriggeredConformalArm:
    def __init__(self, k: float = 0.5, h: float = 4.0,
                 normal_window_days: int = 90, alarm_window_days: int = 21,
                 cooldown_days: int = 14):
        self.k = k
        self.h = h
        self.normal_window_days = normal_window_days
        self.alarm_window_days = alarm_window_days
        self.cooldown_days = cooldown_days

        self.mu_0: float = 0.0
        self.sigma_0: float = 1.0
        self.is_initialized: bool = False

        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self.last_alarm_date: Optional[pd.Timestamp] = None

    def initialize_baseline(self, residual_store, burnin_end: pd.Timestamp, model: str) -> None:
        """
        Calculates baseline mu and sigma strictly from out-of-sample burn-in residuals.
        """
        scores = residual_store.get_observable_residuals(current_origin=burnin_end, model=model)
        if len(scores) > 5:
            self.mu_0 = float(np.mean(scores))
            self.sigma_0 = float(max(1e-4, np.std(scores)))
        else:
            self.mu_0 = 0.2
            self.sigma_0 = 0.1
        self.is_initialized = True

    def update_cusum(self, origin: pd.Timestamp, model: str, residual_store) -> None:
        """
        Updates CUSUM accumulator on the most recent observable residual.
        """
        if not self.is_initialized:
            return

        recent_scores = residual_store.get_observable_residuals(
            current_origin=origin,
            model=model,
            max_window_days=7,  # look at errors observed since last step
        )
        if len(recent_scores) == 0:
            return

        mean_recent = float(np.mean(recent_scores))
        z = (mean_recent - self.mu_0) / self.sigma_0

        self.s_pos = max(0.0, self.s_pos + z - self.k)
        self.s_neg = max(0.0, self.s_neg - z - self.k)

        if self.s_pos > self.h or self.s_neg > self.h:
            self.last_alarm_date = origin
            # Reset CUSUM after alarm
            self.s_pos = 0.0
            self.s_neg = 0.0

    def get_effective_window(self, origin: pd.Timestamp) -> int:
        if self.last_alarm_date is not None:
            days_since_alarm = (origin - self.last_alarm_date).days
            if 0 <= days_since_alarm <= self.cooldown_days:
                return self.alarm_window_days
        return self.normal_window_days

    def get_intervals(self, y_hat: float, origin: pd.Timestamp, horizon: int,
                      level: float, model: str, residual_store) -> Tuple[float, float, float]:
        window = self.get_effective_window(origin)
        scores = residual_store.get_observable_residuals(
            current_origin=origin,
            model=model,
            horizon=horizon,
            max_window_days=window,
        )
        if len(scores) < 10:
            scores = residual_store.get_observable_residuals(
                current_origin=origin,
                model=model,
                max_window_days=window,
            )

        q = conformal_quantile(scores, level)
        return log_bounds_to_raw(y_hat, q)
