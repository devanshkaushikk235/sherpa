"""Arm D: Adaptive Conformal Inference (ACI - Gibbs & Candès 2021).
Dynamically adjusts effective noncoverage level alpha_t based on empirical coverage feedback:
alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
where err_t = 1 if y_t not in interval else 0.

Supports delayed feedback per horizon h:
At origin t, feedback is applied ONLY when realized targets reach origin (t_target <= t),
preventing lookahead leakage in multi-step (h > 1) forecasting.
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class AdaptiveConformalArm:
    def __init__(self, gamma: float = 0.005, window_days: int = 90, min_samples: int = 10):
        self.gamma = gamma
        self.window_days = window_days
        self.min_samples = min_samples
        # Map: (horizon, level) -> alpha_t. horizon=0 represents pooled/global fallback.
        self.effective_alpha: Dict[Tuple[int, float], float] = {}

    def _get_alpha_t(self, nominal_level: float, horizon: Optional[int] = None) -> float:
        h_key = horizon if horizon is not None else 0
        key = (h_key, nominal_level)
        if key not in self.effective_alpha:
            # Fall back to pooled if exists, else initial nominal alpha
            pooled_key = (0, nominal_level)
            self.effective_alpha[key] = self.effective_alpha.get(pooled_key, 1.0 - nominal_level)
        return self.effective_alpha[key]

    def update_feedback(self, nominal_level: float, was_covered: bool,
                        horizon: Optional[int] = None) -> None:
        """
        Updates alpha_t given recent empirical coverage observation whose target has matured.
        """
        nominal_alpha = 1.0 - nominal_level
        err = 0.0 if was_covered else 1.0
        h_key = horizon if horizon is not None else 0
        alpha_t = self._get_alpha_t(nominal_level, horizon=h_key)

        # Gradient update
        alpha_new = alpha_t + self.gamma * (nominal_alpha - err)
        # Clip to valid probability bounds [0.01, 0.99]
        clamped = float(np.clip(alpha_new, 0.01, 0.99))
        self.effective_alpha[(h_key, nominal_level)] = clamped

        # Also update pooled if horizon-specific
        if h_key != 0:
            pooled_alpha = self._get_alpha_t(nominal_level, horizon=0)
            pooled_new = float(np.clip(pooled_alpha + self.gamma * (nominal_alpha - err), 0.01, 0.99))
            self.effective_alpha[(0, nominal_level)] = pooled_new

    def get_intervals(self, y_hat: float, origin: pd.Timestamp, horizon: int,
                      nominal_level: float, model: str, residual_store) -> Tuple[float, float, float]:
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

        q = conformal_quantile(scores, effective_level)
        return log_bounds_to_raw(y_hat, q)
