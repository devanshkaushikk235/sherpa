"""Base utilities for conformal intervals and log-space transformation."""
import numpy as np
from typing import Tuple


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """
    Computes finite-sample conformal quantile:
    q = score at rank ceil((n + 1) * level) / n.
    If n is small (e.g. < 5), falls back to maximum score or a robust default.
    """
    n = len(scores)
    if n == 0:
        return 0.5  # fallback
    k = int(np.ceil((n + 1) * level))
    k = min(n, max(1, k))
    # sort and pick (k - 1) 0-indexed
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def log_bounds_to_raw(y_hat: float, q_log: float) -> Tuple[float, float, float]:
    """
    Transforms log-scale conformal quantile into raw positive player-count bounds:
    lo = max(0, exp(log(y_hat + 1) - q) - 1)
    hi = exp(log(y_hat + 1) + q) - 1
    Returns (lo, hi, width).
    """
    log_hat = np.log1p(max(0.0, y_hat))
    lo = float(max(0.0, np.expm1(log_hat - q_log)))
    hi = float(max(0.0, np.expm1(log_hat + q_log)))
    width = float(hi - lo)
    return lo, hi, width
