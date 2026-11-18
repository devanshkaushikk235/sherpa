"""Arm A: Fixed Conformal Intervals.
Calibrated on out-of-sample residuals from the burn-in period (days 366..455).
Quantiles are frozen once and never updated during subsequent evaluation.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.intervals.base import conformal_quantile, log_bounds_to_raw


class FixedConformalArm:
    def __init__(self, levels: Tuple[float, ...] = (0.8, 0.9)):
        self.levels = levels
        # Map: (horizon, level) -> frozen_q_log
        self.frozen_quantiles: Dict[Tuple[int, float], float] = {}
        self.is_calibrated = False

    def calibrate(self, residual_store, burnin_end: pd.Timestamp, model: str, horizons: range) -> None:
        """
        Extracts out-of-sample residuals observable on or before burnin_end.
        Freezes the conformal quantile for each horizon and target level.
        """
        for h in horizons:
            scores = residual_store.get_observable_residuals(
                current_origin=burnin_end,
                model=model,
                horizon=h,
            )
            # If horizon-specific count is small (<15), pool all observable residuals for the model
            if len(scores) < 15:
                scores = residual_store.get_observable_residuals(
                    current_origin=burnin_end,
                    model=model,
                )

            for lvl in self.levels:
                q = conformal_quantile(scores, lvl)
                self.frozen_quantiles[(h, lvl)] = q

        self.is_calibrated = True

    def get_intervals(self, y_hat: float, horizon: int, level: float) -> Tuple[float, float, float]:
        if not self.is_calibrated:
            raise RuntimeError("FixedConformalArm must be calibrated before generating intervals")
        q = self.frozen_quantiles.get((horizon, level), 0.5)
        return log_bounds_to_raw(y_hat, q)
