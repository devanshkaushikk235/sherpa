"""Arm F: Hybrid Patch-Day Inflation + Exponential ACI Taper.
Combines schedule-aware proactive safety with reactive online adaptation:
1. At patch launch (d = 0 days since patch), applies an initial safety multiplier kappa_0 (e.g. 1.35x).
2. Over days 1..patch_window_days, decays the multiplier exponentially:
       kappa(d) = 1.0 + (kappa_0 - 1.0) * exp(-d / tau)
3. Concurrently, ACI tracks empirical coverage errors via delayed feedback,
   ensuring the interval seamlessly settles at the calibrated nominal level without
   incurring the prolonged 40% interval width penalty of static scheduled inflation.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class HybridConformalArm:
    def __init__(
        self,
        gamma: float = 0.005,
        window_days: int = 90,
        patch_window_days: int = 7,
        inflation_factor: float = 1.35,
        decay_tau: float = 2.5,
        min_samples: int = 10,
    ):
        self.gamma = gamma
        self.window_days = window_days
        self.patch_window_days = patch_window_days
        self.inflation_factor = inflation_factor
        self.decay_tau = decay_tau
        self.min_samples = min_samples

        # Map: (horizon, level) -> alpha_t. horizon=0 represents pooled/global fallback.
        self.effective_alpha: Dict[Tuple[int, float], float] = {}

    def _get_alpha_t(self, nominal_level: float, horizon: Optional[int] = None) -> float:
        h_key = horizon if horizon is not None else 0
        key = (h_key, nominal_level)
        if key not in self.effective_alpha:
            pooled_key = (0, nominal_level)
            self.effective_alpha[key] = self.effective_alpha.get(pooled_key, 1.0 - nominal_level)
        return self.effective_alpha[key]

    def update_feedback(
        self, nominal_level: float, was_covered: bool, horizon: Optional[int] = None
    ) -> None:
        """
        Updates alpha_t given recent empirical coverage observation whose target has matured.
        """
        nominal_alpha = 1.0 - nominal_level
        err = 0.0 if was_covered else 1.0
        h_key = horizon if horizon is not None else 0
        alpha_t = self._get_alpha_t(nominal_level, horizon=h_key)

        alpha_new = alpha_t + self.gamma * (nominal_alpha - err)
        clamped = float(np.clip(alpha_new, 0.01, 0.99))
        self.effective_alpha[(h_key, nominal_level)] = clamped

        if h_key != 0:
            pooled_alpha = self._get_alpha_t(nominal_level, horizon=0)
            pooled_new = float(np.clip(pooled_alpha + self.gamma * (nominal_alpha - err), 0.01, 0.99))
            self.effective_alpha[(0, nominal_level)] = pooled_new

    def get_patch_multiplier(
        self, target_date: pd.Timestamp, origin: pd.Timestamp, patches_df: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculates exponential taper multiplier kappa(d) based on days since most recent patch:
        kappa(d) = 1.0 + (inflation_factor - 1.0) * exp(-d / tau)
        Returns 1.0 if not within patch window.
        """
        if patches_df is None or patches_df.empty or "date" not in patches_df.columns:
            return 1.0

        p_dates = pd.to_datetime(patches_df["date"].values)
        td = pd.to_datetime(target_date)
        orig = pd.to_datetime(origin)

        # Case 1: A patch is scheduled between origin and target_date (imminent launch)
        intervening = (p_dates > orig) & (p_dates <= td)
        if intervening.any():
            return float(self.inflation_factor)

        # Case 2: Target date is within patch_window_days of a past patch
        past_patches = p_dates[p_dates <= td]
        if len(past_patches) == 0:
            return 1.0

        diffs_days = (td - past_patches).total_seconds() / 86400.0
        min_diff = float(diffs_days.min())

        if 0.0 <= min_diff <= float(self.patch_window_days):
            # Exponential decay towards 1.0
            excess = (self.inflation_factor - 1.0) * np.exp(-min_diff / self.decay_tau)
            return float(1.0 + excess)

        return 1.0

    def get_intervals(
        self,
        y_hat: float,
        origin: pd.Timestamp,
        horizon: int,
        nominal_level: float,
        model: str,
        residual_store,
        patches_df: Optional[pd.DataFrame] = None,
        return_base: bool = False,
    ):
        alpha_t = self._get_alpha_t(nominal_level, horizon=horizon)
        effective_level = 1.0 - alpha_t

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

        q_base = conformal_quantile(scores, effective_level)

        # Apply exponential patch taper
        target_date = origin + pd.Timedelta(days=horizon)
        multiplier = self.get_patch_multiplier(target_date, origin, patches_df)
        q_inflated = float(q_base * multiplier)

        lo, hi, width = log_bounds_to_raw(y_hat, q_inflated)
        if return_base:
            lo_base, hi_base, _ = log_bounds_to_raw(y_hat, q_base)
            return lo, hi, width, lo_base, hi_base
        return lo, hi, width
