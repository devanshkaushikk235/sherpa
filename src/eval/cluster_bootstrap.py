"""Cluster Bootstrap over Patch Events.
Resamples patch event clusters with replacement, stratified by game.
Computes paired difference distributions for all key arm pairs across:
- Coverage difference (0-7d)
- Width difference (0-7d)
- Winkler difference (0-7d)
- Capacity Cost @ 5x difference (0-7d)
Never averages CI endpoints across games.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


PAIR_COMPARISONS = [
    ("rolling", "fixed"),
    ("triggered", "fixed"),
    ("aci", "fixed"),
    ("scheduled", "fixed"),
    ("hybrid", "fixed"),
    ("hybrid", "scheduled"),
    ("hybrid", "aci"),       # Pre-registered primary comparison
    ("hybrid", "rolling"),
    ("aci", "rolling"),
    ("triggered", "rolling"),
]


def run_cluster_bootstrap(
    patch_level_df: pd.DataFrame,
    n_resamples: int = 2000,
    seed: int = 42,
    pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Executes stratified cluster bootstrap over patch events.
    Computes both pooled (cross-game) and per-game paired confidence intervals.
    """
    if pairs is None:
        pairs = PAIR_COMPARISONS

    if patch_level_df.empty or "model" not in patch_level_df.columns:
        return pd.DataFrame(columns=[
            "scope", "model", "level", "arm_x", "arm_y", "is_primary", "n_patches",
            "cov_diff_mean", "cov_diff_ci_lo", "cov_diff_ci_hi", "cov_p_val", "cov_p_str",
            "width_diff_mean", "width_diff_ci_lo", "width_diff_ci_hi", "width_p_val", "width_p_str",
            "norm_width_diff_mean", "norm_width_diff_ci_lo", "norm_width_diff_ci_hi",
            "winkler_diff_mean", "winkler_diff_ci_lo", "winkler_diff_ci_hi", "winkler_p_val",
            "cost5x_diff_mean", "cost5x_diff_ci_lo", "cost5x_diff_ci_hi", "cost5x_p_val", "cost5x_p_str",
        ])

    rng = np.random.default_rng(seed)
    results = []

    models = patch_level_df["model"].unique()
    levels = patch_level_df["level"].unique()

    for model in models:
        for lvl in levels:
            sub = patch_level_df[
                (patch_level_df["model"] == model) &
                (patch_level_df["level"] == lvl)
            ]

            games = sub["game_id"].unique()

            # We evaluate: (1) Pooled across games, (2) Per individual game
            eval_scopes = [("pooled", games)] + [(gid, [gid]) for gid in games]

            for scope_name, scope_games in eval_scopes:
                scope_df = sub[sub["game_id"].isin(scope_games)]

                # Build dictionary of patches per game
                game_patches = {
                    gid: sorted(scope_df[scope_df["game_id"] == gid]["patch_id"].unique())
                    for gid in scope_games
                }
                total_patches = sum(len(p) for p in game_patches.values())

                if total_patches < 3:
                    continue

                for arm_x, arm_y in pairs:
                    # Merge on (game_id, patch_id)
                    px = scope_df[scope_df["arm"] == arm_x]
                    py = scope_df[scope_df["arm"] == arm_y]

                    merged = pd.merge(
                        px, py,
                        on=["game_id", "patch_id"],
                        suffixes=("_x", "_y")
                    )

                    if len(merged) < 3:
                        continue

                    # Precompute paired arrays
                    cov_diff_arr = merged["coverage_x"].values - merged["coverage_y"].values
                    width_diff_arr = merged["mean_width_x"].values - merged["mean_width_y"].values
                    has_norm_w = "norm_width_x" in merged.columns and "norm_width_y" in merged.columns
                    norm_w_diff_arr = (merged["norm_width_x"].values - merged["norm_width_y"].values) if has_norm_w else np.zeros_like(cov_diff_arr)
                    wink_diff_arr = merged["winkler_x"].values - merged["winkler_y"].values
                    cost5_diff_arr = merged["cost_5x_x"].values - merged["cost_5x_y"].values

                    # Map game_id and patch_id to indices in merged
                    patch_idx_map = {}
                    for i, r in merged.iterrows():
                        key = (r["game_id"], r["patch_id"])
                        patch_idx_map[key] = i

                    # Bootstrap resampling loop
                    boot_cov, boot_width, boot_norm_w, boot_wink, boot_cost5 = [], [], [], [], []

                    for _ in range(n_resamples):
                        sampled_indices = []
                        # Resample patches with replacement within each game (stratified)
                        for gid in scope_games:
                            p_list = game_patches[gid]
                            n_p = len(p_list)
                            if n_p == 0:
                                continue
                            sampled_p = rng.choice(p_list, size=n_p, replace=True)
                            for p in sampled_p:
                                idx = patch_idx_map.get((gid, p))
                                if idx is not None:
                                    sampled_indices.append(idx)

                        if not sampled_indices:
                            continue

                        s_idx = np.array(sampled_indices)
                        boot_cov.append(float(np.mean(cov_diff_arr[s_idx])))
                        boot_width.append(float(np.mean(width_diff_arr[s_idx])))
                        boot_norm_w.append(float(np.mean(norm_w_diff_arr[s_idx])))
                        boot_wink.append(float(np.mean(wink_diff_arr[s_idx])))
                        boot_cost5.append(float(np.mean(cost5_diff_arr[s_idx])))

                    if not boot_cov:
                        continue

                    def get_stats(arr):
                        mean_val = float(np.mean(arr))
                        lo_ci = float(np.percentile(arr, 2.5))
                        hi_ci = float(np.percentile(arr, 97.5))
                        # Empirical two-sided p-value vs zero diff
                        p_val = float(2.0 * min(np.mean(np.array(arr) <= 0), np.mean(np.array(arr) >= 0)))
                        p_val = min(1.0, max(1.0 / n_resamples, p_val))
                        p_str = "< 0.001" if p_val <= (1.0 / n_resamples) else f"{p_val:.3f}"
                        return mean_val, lo_ci, hi_ci, p_val, p_str

                    c_m, c_lo, c_hi, c_p, c_p_str = get_stats(boot_cov)
                    w_m, w_lo, w_hi, w_p, w_p_str = get_stats(boot_width)
                    nw_m, nw_lo, nw_hi, nw_p, nw_p_str = get_stats(boot_norm_w)
                    k_m, k_lo, k_hi, k_p, k_p_str = get_stats(boot_wink)
                    cost_m, cost_lo, cost_hi, cost_p, cost_p_str = get_stats(boot_cost5)

                    results.append({
                        "scope": scope_name,
                        "model": model,
                        "level": float(lvl),
                        "arm_x": arm_x,
                        "arm_y": arm_y,
                        "is_primary": bool(arm_x == "hybrid" and arm_y == "aci"),
                        "n_patches": len(merged),
                        # Coverage difference
                        "cov_diff_mean": c_m,
                        "cov_diff_ci_lo": c_lo,
                        "cov_diff_ci_hi": c_hi,
                        "cov_p_val": c_p,
                        "cov_p_str": c_p_str,
                        # Width difference
                        "width_diff_mean": w_m,
                        "width_diff_ci_lo": w_lo,
                        "width_diff_ci_hi": w_hi,
                        "width_p_val": w_p,
                        "width_p_str": w_p_str,
                        # Normalized width difference
                        "norm_width_diff_mean": nw_m,
                        "norm_width_diff_ci_lo": nw_lo,
                        "norm_width_diff_ci_hi": nw_hi,
                        # Winkler difference
                        "winkler_diff_mean": k_m,
                        "winkler_diff_ci_lo": k_lo,
                        "winkler_diff_ci_hi": k_hi,
                        "winkler_p_val": k_p,
                        # Cost 5x difference
                        "cost5x_diff_mean": cost_m,
                        "cost5x_diff_ci_lo": cost_lo,
                        "cost5x_diff_ci_hi": cost_hi,
                        "cost5x_p_val": cost_p,
                        "cost5x_p_str": cost_p_str,
                    })

    return pd.DataFrame(results)
