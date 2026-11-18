"""Direct Multi-Horizon LightGBM Forecasting.
Trains 14 separate models (h = 1..14) using historical origins tau where tau + h <= origin.
Strictly prevents error compounding and maintains clean, horizon-specific residual streams.
"""
from typing import Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd


def extract_features_at_origin(series_df: pd.DataFrame, origin: pd.Timestamp,
                               lags: List[int] = [0, 1, 2, 6, 13, 20, 27],
                               rolling_windows: List[int] = [7, 14, 28]) -> Optional[Dict[str, float]]:
    """
    Extracts lag and rolling features from historical y_peak on or before origin.
    Returns None if insufficient history is available.
    """
    history = series_df[series_df["date"] <= origin].sort_values("date")
    max_lag = max(lags)
    max_window = max(rolling_windows)
    required_days = max(max_lag, max_window) + 1
    if len(history) < required_days:
        return None

    # Get values aligned by date
    y_vals = history["y_peak"].values
    if np.isnan(y_vals[-1]):
        # if the immediate origin value is missing, interpolate or use last valid
        valid = y_vals[~np.isnan(y_vals)]
        if len(valid) == 0:
            return None
        last_val = valid[-1]
    else:
        last_val = y_vals[-1]

    # Convert to log1p for stability
    log_y = np.log1p(np.maximum(0.0, np.nan_to_num(y_vals, nan=last_val)))

    feat = {}
    for lag in lags:
        idx = -(lag + 1)
        feat[f"lag_{lag}"] = log_y[idx] if abs(idx) <= len(log_y) else log_y[0]

    for w in rolling_windows:
        w_slice = log_y[-w:]
        feat[f"roll_mean_{w}"] = float(np.mean(w_slice))
        feat[f"roll_std_{w}"] = float(np.std(w_slice))

    return feat


def calendar_features_for_date(d: pd.Timestamp) -> Dict[str, float]:
    return {
        "target_dow": float(d.dayofweek),
        "target_day": float(d.day),
        "target_month": float(d.month),
        "target_is_weekend": float(1.0 if d.dayofweek in (5, 6) else 0.0),
    }


class DirectLightGBMForecaster:
    def __init__(self, horizon: int = 14,
                 lags: List[int] = [0, 1, 2, 6, 13, 20, 27],
                 rolling_windows: List[int] = [7, 14, 28],
                 params: Optional[dict] = None):
        self.horizon = horizon
        self.lags = lags
        self.rolling_windows = rolling_windows
        self.params = params or {
            "n_estimators": 80,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 10,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,
        }
        self.models: Dict[int, lgb.LGBMRegressor] = {}

    def fit_and_predict(self, history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
        """
        Fits 14 direct models on all valid historical origins tau where tau + h <= origin.
        Predicts y_hat for horizons 1..horizon (target dates origin + 1..origin + horizon).
        """
        # History strictly <= origin
        df = history[history["date"] <= origin].sort_values("date").reset_index(drop=True)
        date_to_idx = {d: i for i, d in enumerate(df["date"])}
        y_vals = df["y_peak"].values

        # Build feature matrix for candidate historical origins
        # Origin candidate tau must have enough history before it
        min_tau_idx = max(max(self.lags), max(self.rolling_windows)) + 1
        n_dates = len(df)
        
        # Vectorized precomputation of lag and rolling features across all history up to origin
        valid_y = np.maximum(0.0, np.nan_to_num(y_vals, nan=y_vals[-1] if len(y_vals) > 0 else 1.0))
        log_y = np.log1p(valid_y)
        s_log = pd.Series(log_y)

        roll_means = {w: s_log.rolling(w).mean().values for w in self.rolling_windows}
        roll_stds = {w: s_log.rolling(w).std().values for w in self.rolling_windows}

        origin_feats = {}
        for idx in range(min_tau_idx, n_dates):
            feat = {}
            for lag in self.lags:
                pos = idx - lag
                feat[f"lag_{lag}"] = float(log_y[pos]) if pos >= 0 else float(log_y[0])
            for w in self.rolling_windows:
                m_val = roll_means[w][idx]
                s_val = roll_stds[w][idx]
                feat[f"roll_mean_{w}"] = float(m_val) if not np.isnan(m_val) else 0.0
                feat[f"roll_std_{w}"] = float(s_val) if not np.isnan(s_val) else 0.0
            origin_feats[idx] = feat

        # Feature vector at current origin (t) is the feature dict at index n_dates - 1
        current_feat = origin_feats.get(n_dates - 1)
        if current_feat is None:
            current_feat = extract_features_at_origin(df, origin, self.lags, self.rolling_windows)

        results = []
        for h in range(1, self.horizon + 1):
            target_date = origin + pd.Timedelta(days=h)
            cal_feat = calendar_features_for_date(target_date)

            # Training set for horizon h:
            # tau can range up to (n_dates - 1 - h), so tau + h <= origin
            X_train, y_train = [], []
            for idx, feat_dict in origin_feats.items():
                target_idx = idx + h
                if target_idx < n_dates:
                    target_y = y_vals[target_idx]
                    if not np.isnan(target_y) and target_y > 0:
                        row = dict(feat_dict)
                        tau_target_date = df.at[target_idx, "date"]
                        row.update(calendar_features_for_date(tau_target_date))
                        X_train.append(row)
                        y_train.append(np.log1p(target_y))

            if len(X_train) < 30:
                # If too few samples, fallback to seasonal naive
                k = int(np.ceil(h / 7))
                ref_date = target_date - pd.Timedelta(days=k * 7)
                ref_val = df.loc[df["date"] == ref_date, "y_peak"].values
                y_hat = float(ref_val[0]) if len(ref_val) > 0 and not np.isnan(ref_val[0]) else float(np.nanmedian(y_vals))
            else:
                X_df = pd.DataFrame(X_train)
                y_arr = np.array(y_train)
                model = lgb.LGBMRegressor(**self.params)
                model.fit(X_df, y_arr)
                self.models[h] = model

                # Predict for current origin
                test_row = dict(current_feat)
                test_row.update(cal_feat)
                pred_log = float(model.predict(pd.DataFrame([test_row]))[0])
                y_hat = float(max(1.0, np.expm1(pred_log)))

            results.append({
                "horizon": h,
                "target_date": target_date,
                "y_hat": y_hat,
            })

        return pd.DataFrame(results)
