"""Arm E: Schedule-Aware Conformal Intervals.
Baseline that leverages known/scheduled patch dates to widen intervals or inflate quantiles.
At known patch dates (target_date within patch_window_days of a patch), inflates the conformal quantile
by a configurable inflation factor (default 1.35).
This is the schedule-informed baseline that reactive methods (CUSUM, ACI) must compete against.
"""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class ScheduledConformalArm:
    def __init__(self, window_days: int = 90, patch_window_days: int = 7,
                 inflation_factor: float = 1.35, min_samples: int = 10):
        self.window_days = window_days
        self.patch_window_days = patch_window_days
        self.inflation_factor = inflation_factor
        self.min_samples = min_samples

    def is_patch_affected(self, target_date: pd.Timestamp,
                          origin: pd.Timestamp,
                          patches_df: Optional[pd.DataFrame]) -> bool:
        """
        Returns True iff target_date itself is within patch_window_days of the nearest
        patch on or before target_date.

        Does NOT flag a target day simply because a patch falls somewhere in the forecast
        horizon — only the specific target_date's own proximity to a past patch is tested.
        This prevents over-inflation of non-affected days in multi-step (h > 1) forecasts.
        """
        if patches_df is None or patches_df.empty or "date" not in patches_df.columns:
            return False

        p_dates = pd.to_datetime(patches_df["date"].values)
        td = pd.to_datetime(target_date)

        # Only consider patches already on or before target_date
        past_patches = p_dates[p_dates <= td]
        if len(past_patches) == 0:
            return False

        diffs = (td - past_patches).days
        return bool(diffs.min() <= self.patch_window_days)


    def get_intervals(self, y_hat: float, origin: pd.Timestamp, horizon: int,
                      level: float, model: str, residual_store,
                      patches_df: Optional[pd.DataFrame] = None) -> Tuple[float, float, float]:
        """
        Computes conformal prediction interval, inflating quantile if target_date is patch-affected.
        """
        target_date = origin + pd.Timedelta(days=horizon)
        patch_affected = self.is_patch_affected(target_date, origin, patches_df)

        scores = residual_store.get_observable_residuals(
            current_origin=origin,
            model=model,
            horizon=horizon,
            max_window_days=self.window_days,
        )
        if len(scores) < self.min_samples:
            scores = residual_store.get_observable_residuals(
                current_origin=origin,
                model=model,
                max_window_days=self.window_days,
            )

        q = conformal_quantile(scores, level)

        if patch_affected:
            q = float(q * self.inflation_factor)

        return log_bounds_to_raw(y_hat, q)
