"""Illustrative Asymmetric Server Capacity Cost Sensitivity Analysis.
Evaluates the upper interval bound hi as an autoscaling provisioning policy:
Cost_r = sum [ r * max(0, y - hi) + 1.0 * max(0, hi - y) ] / (n * mean_y)
Tested over penalty ratios r in [2.0, 5.0, 10.0].
Framed strictly as an illustrative sensitivity analysis.
"""
from typing import Dict, List
import numpy as np
import pandas as pd


def compute_capacity_cost_sensitivity(intervals_df: pd.DataFrame,
                                     ratios: List[float] = [2.0, 5.0, 10.0]) -> Dict[float, float]:
    """
    Computes normalized asymmetric capacity cost across specified penalty ratios r.
    r represents the penalty multiplier for under-provisioning (outage/queue risk)
    relative to 1.0 for over-provisioning (idle instance cost).
    """
    valid = intervals_df.dropna(subset=["hi", "y_true"]).copy()
    valid = valid[valid["y_true"] > 0]
    if len(valid) == 0:
        return {r: np.nan for r in ratios}

    y = valid["y_true"].values
    hi = valid["hi"].values
    mean_y = float(np.mean(y))

    under = np.maximum(0.0, y - hi)   # under-provisioning
    over = np.maximum(0.0, hi - y)    # over-provisioning

    costs = {}
    for r in ratios:
        total_loss = float(np.sum(r * under + 1.0 * over))
        norm_cost = total_loss / (len(valid) * max(1.0, mean_y))
        costs[float(r)] = float(norm_cost)

    return costs


def compute_break_even_penalty_ratio(y_true: np.ndarray, hi_arm: np.ndarray,
                                     hi_base: np.ndarray) -> float:
    """
    Computes the break-even penalty ratio r* at which hi_arm achieves equal total cost
    to hi_base under asymmetric loss: Cost_r = r * Under + 1.0 * Over.
    
    r* = (Over_arm - Over_base) / (Under_base - Under_arm)
    
    If Under_base <= Under_arm (arm fails to reduce under-provisioning),
    returns np.inf (arm never breaks even).
    If Over_arm <= Over_base (arm has equal or less idle capacity while reducing under),
    returns 0.0 (arm dominates at all r >= 0).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(hi_arm) & ~np.isnan(hi_base) & (y_true > 0)
    if not np.any(mask):
        return np.nan

    y = y_true[mask]
    hi_x = hi_arm[mask]
    hi_b = hi_base[mask]

    under_x = np.sum(np.maximum(0.0, y - hi_x))
    under_b = np.sum(np.maximum(0.0, y - hi_b))

    over_x = np.sum(np.maximum(0.0, hi_x - y))
    over_b = np.sum(np.maximum(0.0, hi_b - y))

    delta_under = under_b - under_x
    delta_over = over_x - over_b

    if delta_under <= 0:
        # Arm has more or equal shortfall than base; can never beat base
        return float(np.inf)

    if delta_over <= 0:
        # Arm has strictly less idle capacity and less shortfall; dominates everywhere
        return 0.0

    r_star = float(delta_over / delta_under)
    return r_star
