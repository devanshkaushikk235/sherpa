"""Publication-ready figure generation for writeup and presentation.
1. fig1_coverage_by_window_ci.png     — Coverage by window x arm, nominal line, N
2. fig2_width_coverage_frontier.png   — Width vs Coverage frontier per arm in 0-7d window
3. fig3_cost_sensitivity_breakeven.png— Cost vs Penalty Ratio curves, break-even points
4. fig4_event_study_days_since_patch.png — Coverage AND width trajectory 0-14d post-patch
5. fig5_hybrid_sensitivity_heatmap.png— Sensitivity heat map (kappa0 x tau)
6. fig6_per_game_small_multiples.png  — Small multiples of post-patch degradation per game
7. fig7_reliability_diagram.png       — Calibration plot: empirical vs nominal coverage
8. fig8_point_forecast_degradation.png— MAPE by window: model-bias vs calibration failure
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARM_COLORS = {
    "fixed":     "#6c757d",   # Gray
    "rolling":   "#28a745",   # Green
    "triggered": "#fd7e14",   # Orange
    "aci":       "#007bff",   # Blue
    "scheduled": "#dc3545",   # Red
    "hybrid":    "#6f42c1",   # Purple
}

ARM_LABELS = {
    "fixed":     "Fixed (Arm A)",
    "rolling":   "Rolling 90d (Arm B)",
    "triggered": "CUSUM (Arm C)",
    "aci":       "ACI (Arm D)",
    "scheduled": "Scheduled 1.25x (Arm E)",
    "hybrid":    "Hybrid Taper (Arm F)",
}


def set_plot_style():
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams.update({
        "font.family":    "sans-serif",
        "font.size":      9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize":8,
        "ytick.labelsize":8,
        "legend.fontsize":8,
        "figure.titlesize":11,
        "figure.dpi":     200,
    })


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def plot_fig1_coverage_by_window(metrics_df: pd.DataFrame, out_path: Path, level: float = 0.9):
    """Figure 1: Coverage by window stratum across all arms with nominal target line."""
    set_plot_style()
    sub = metrics_df[(metrics_df["level"] == level) & (metrics_df["model"] == "lgbm")].copy()
    if sub.empty:
        return

    windows    = ["post_patch (0-7d)", "post_patch (8-14d)", "post_patch (15-30d)", "baseline (>30d)"]
    win_labels = ["0-7d (Shock)", "8-14d (Recovery)", "15-30d (Late)", "Baseline (>30d)"]
    arms       = ["fixed", "rolling", "triggered", "aci", "scheduled", "hybrid"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x         = np.arange(len(windows))
    n_arms    = len(arms)
    bar_width = 0.13

    for idx, arm in enumerate(arms):
        arm_sub = sub[sub["arm"] == arm].groupby("window")["coverage"].mean().reindex(windows)
        pos = x + (idx - n_arms / 2.0 + 0.5) * bar_width
        ax.bar(pos, arm_sub * 100, bar_width,
               label=ARM_LABELS.get(arm, arm),
               color=ARM_COLORS.get(arm, "gray"), alpha=0.9)

    ax.axhline(level * 100, color="black", linestyle="--", lw=1.2,
               label=f"Nominal Target ({int(level*100)}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(win_labels)
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_ylim(75, 100)
    ax.set_title("Empirical Coverage by Window Stratum Across Recalibration Arms (EVAL Split, N=211 Patches)")
    ax.legend(loc="lower right", frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def plot_fig2_width_coverage_frontier(metrics_df: pd.DataFrame, out_path: Path, level: float = 0.9):
    """Figure 2: Width vs Coverage Pareto Frontier in 0-7d Post-Patch Window."""
    set_plot_style()
    sub = metrics_df[
        (metrics_df["level"] == level) &
        (metrics_df["model"] == "lgbm") &
        (metrics_df["window"] == "post_patch (0-7d)")
    ].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    agg = sub.groupby("arm")[["coverage", "mean_width"]].mean()

    for arm in ["fixed", "rolling", "triggered", "aci", "scheduled", "hybrid"]:
        if arm not in agg.index:
            continue
        row = agg.loc[arm]
        ax.scatter(row["mean_width"] / 1000.0, row["coverage"] * 100,
                   color=ARM_COLORS.get(arm, "black"), s=120, zorder=5,
                   label=ARM_LABELS.get(arm, arm))
        ax.annotate(f" {arm.title()}",
                    (row["mean_width"] / 1000.0, row["coverage"] * 100), fontsize=8)

    ax.axhline(level * 100, color="black", linestyle="--", lw=1.2,
               label=f"Nominal Target ({int(level*100)}%)")
    ax.set_xlabel("Mean Interval Width (Thousands of Players)")
    ax.set_ylabel("Empirical Coverage (%)")
    ax.set_title("Calibration vs. Sharpness Frontier (0-7d Post-Patch Window)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def plot_fig3_cost_sensitivity_breakeven(intervals_df: pd.DataFrame, out_path: Path, level: float = 0.9):
    """Figure 3: Cost Sensitivity Curves and Break-Even Ratio Analysis."""
    set_plot_style()
    eval_sub = intervals_df[
        (intervals_df["split"] == "EVAL") &
        (intervals_df["level"] == level) &
        (intervals_df["model"] == "lgbm")
    ].copy()
    if eval_sub.empty:
        return

    ratios = np.linspace(1.0, 55.0, 50)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    y      = eval_sub["y_true"].values
    mean_y = float(np.mean(y[y > 0]))

    for arm in ["fixed", "aci", "scheduled", "hybrid"]:
        sub_arm = eval_sub[eval_sub["arm"] == arm]
        under   = np.maximum(0.0, sub_arm["y_true"].values - sub_arm["hi"].values)
        over    = np.maximum(0.0, sub_arm["hi"].values - sub_arm["y_true"].values)
        m_u, m_o = np.mean(under), np.mean(over)
        costs = [(r * m_u + m_o) / mean_y for r in ratios]
        ax.plot(ratios, costs, color=ARM_COLORS.get(arm, "black"), lw=1.8,
                label=ARM_LABELS.get(arm, arm))

    ax.axvline(48.5, color="#6f42c1", linestyle=":", lw=1.2,
               label="Hybrid Break-Even (r*=48.5x)")
    ax.set_xlabel("Under-Provisioning Penalty Ratio (c_under / c_over)")
    ax.set_ylabel("Normalized Capacity Cost")
    ax.set_title("Capacity Loss vs. Penalty Ratio (Fixed is Cheapest until r* > 45x)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4
# ---------------------------------------------------------------------------

def plot_fig4_event_study(intervals_df: pd.DataFrame, out_path: Path, level: float = 0.9):
    """Figure 4: Event-Study Plot — Coverage AND Width trajectory, days 0-14 post-patch.

    Showing width alongside coverage lets readers determine whether coverage collapses
    because the intervals are too narrow or because actuals spike unexpectedly.
    """
    set_plot_style()
    eval_sub = intervals_df[
        (intervals_df["split"] == "EVAL") &
        (intervals_df["level"] == level) &
        (intervals_df["model"] == "lgbm") &
        (intervals_df["in_patch_window"] == True) &
        (intervals_df["days_since_patch"] >= 0) &
        (intervals_df["days_since_patch"] <= 14)
    ].copy()
    if eval_sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    days = np.arange(0, 15)

    for arm in ["fixed", "aci", "scheduled", "hybrid"]:
        sub_arm      = eval_sub[eval_sub["arm"] == arm]
        cov_by_day   = sub_arm.groupby("days_since_patch")["covered"].mean().reindex(days) * 100
        width_by_day = sub_arm.groupby("days_since_patch")["width"].mean().reindex(days) / 1000.0

        ax1.plot(days, cov_by_day, marker="o", ms=4,
                 color=ARM_COLORS.get(arm, "black"), lw=1.5,
                 label=ARM_LABELS.get(arm, arm))
        ax2.plot(days, width_by_day, marker="s", ms=4,
                 color=ARM_COLORS.get(arm, "black"), lw=1.5,
                 label=ARM_LABELS.get(arm, arm))

    ax1.axhline(level * 100, color="black", linestyle="--", lw=1.2,
                label=f"Nominal ({int(level*100)}%)")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_ylim(75, 100)
    ax1.set_title("Post-Patch Event Study: Coverage and Width Decay Trajectory (Days 0–14)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Mean Width (k players)")
    ax2.set_xlabel("Days Since Patch (d = 0 is Patch Launch Day)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5
# ---------------------------------------------------------------------------

def plot_fig5_sensitivity_heatmap(sens_df: pd.DataFrame, out_path: Path):
    """Figure 5: Hybrid Sensitivity Grid Heatmap for (kappa0 x tau)."""
    set_plot_style()
    if sens_df.empty:
        return

    piv_cov   = sens_df.pivot(index="kappa0", columns="tau", values="cov_0_7d") * 100
    piv_width = sens_df.pivot(index="kappa0", columns="tau", values="mean_width_0_7d") / 1000.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.imshow(piv_cov.values, cmap="Blues", origin="lower", aspect="auto", vmin=90, vmax=96)
    ax1.set_xticks(range(len(piv_cov.columns)))
    ax1.set_xticklabels([f"{c}d" for c in piv_cov.columns])
    ax1.set_yticks(range(len(piv_cov.index)))
    ax1.set_yticklabels([f"{r}x" for r in piv_cov.index])
    ax1.set_xlabel("Decay Time Constant (tau)")
    ax1.set_ylabel("Initial Multiplier (kappa0)")
    ax1.set_title("0-7d Coverage (%) [Tuned: 1.25x, 2.0d]")
    for i in range(len(piv_cov.index)):
        for j in range(len(piv_cov.columns)):
            ax1.text(j, i, f"{piv_cov.values[i, j]:.1f}%", ha="center", va="center",
                     color="black" if piv_cov.values[i, j] < 94 else "white", fontsize=8)

    ax2.imshow(piv_width.values, cmap="Oranges", origin="lower", aspect="auto")
    ax2.set_xticks(range(len(piv_width.columns)))
    ax2.set_xticklabels([f"{c}d" for c in piv_width.columns])
    ax2.set_yticks(range(len(piv_width.index)))
    ax2.set_yticklabels([f"{r}x" for r in piv_width.index])
    ax2.set_xlabel("Decay Time Constant (tau)")
    ax2.set_title("0-7d Mean Width (k players)")
    for i in range(len(piv_width.index)):
        for j in range(len(piv_width.columns)):
            ax2.text(j, i, f"{piv_width.values[i, j]:.0f}k", ha="center", va="center",
                     color="black" if piv_width.values[i, j] < 200 else "white", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6
# ---------------------------------------------------------------------------

def plot_fig6_per_game_small_multiples(metrics_df: pd.DataFrame, out_path: Path, level: float = 0.9):
    """Figure 6: Per-Game Small Multiples for Post-Patch Degradation."""
    set_plot_style()
    sub = metrics_df[(metrics_df["level"] == level) & (metrics_df["model"] == "lgbm")].copy()
    if sub.empty:
        return

    games     = sorted(sub["game_id"].unique())
    fig, axes = plt.subplots(1, len(games), figsize=(3.5 * len(games), 3.5), sharey=True)
    if len(games) == 1:
        axes = [axes]

    arms      = ["fixed", "rolling", "aci", "scheduled", "hybrid"]
    windows   = ["post_patch (0-7d)", "baseline (>30d)"]
    x         = np.arange(len(windows))
    bar_width = 0.15

    for ax, gid in zip(axes, games):
        g_sub = sub[sub["game_id"] == gid]
        for idx, arm in enumerate(arms):
            vals = (
                g_sub[g_sub["arm"] == arm]
                .groupby("window")["coverage"]
                .mean()
                .reindex(windows) * 100
            )
            pos = x + (idx - len(arms) / 2.0 + 0.5) * bar_width
            ax.bar(pos, vals, bar_width,
                   label=ARM_LABELS.get(arm, arm) if gid == games[0] else "",
                   color=ARM_COLORS.get(arm, "gray"))

        ax.axhline(level * 100, color="black", linestyle="--", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["0-7d (Shock)", "Baseline (>30d)"])
        ax.set_title(gid.replace("_", " ").title())
        ax.set_ylim(70, 100)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Empirical Coverage (%)")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — NEW: Reliability Diagram
# ---------------------------------------------------------------------------

def plot_fig7_reliability_diagram(metrics_df: pd.DataFrame, out_path: Path):
    """Figure 7: Reliability Diagram (Calibration Plot).

    Plots empirical coverage vs nominal level (0.80, 0.90, 0.95) for each arm,
    stratified by post-patch window vs baseline.  A perfectly calibrated arm lies
    on the diagonal.  Under-coverage appears below the diagonal; over-coverage above.

    This is the standard figure in conformal prediction papers and immediately
    communicates the shape of miscalibration without requiring table reading.
    """
    set_plot_style()
    if metrics_df.empty or "level" not in metrics_df.columns:
        return

    sub = metrics_df[metrics_df["model"] == "lgbm"].copy()
    windows_of_interest = {
        "post_patch (0-7d)": "0-7d (Shock)",
        "baseline (>30d)":   "Baseline (>30d)",
    }
    arms = ["fixed", "rolling", "triggered", "aci", "scheduled", "hybrid"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, (win_key, win_label) in zip(axes, windows_of_interest.items()):
        win_sub = sub[sub["window"] == win_key]
        if win_sub.empty:
            ax.set_title(f"{win_label} — no data")
            continue

        # Perfect-calibration diagonal
        diag = np.array([0.75, 1.0])
        ax.plot(diag, diag, "k--", lw=1.2, label="Perfect calibration", zorder=1)
        # Shade under/over-coverage regions
        ax.fill_between(diag, diag, 1.0, alpha=0.04, color="green", label="Over-coverage zone")
        ax.fill_between(diag, 0.75, diag, alpha=0.04, color="red",   label="Under-coverage zone")

        for arm in arms:
            arm_sub = win_sub[win_sub["arm"] == arm].groupby("level")["coverage"].mean()
            if arm_sub.empty:
                continue
            levels_x = arm_sub.index.values
            covs_y   = arm_sub.values
            ax.plot(levels_x, covs_y, marker="o", ms=5,
                    color=ARM_COLORS.get(arm, "black"), lw=1.5,
                    label=ARM_LABELS.get(arm, arm), zorder=3)
            ax.annotate(arm, (levels_x[-1], covs_y[-1]), fontsize=7,
                        xytext=(3, 0), textcoords="offset points")

        ax.set_xlim(0.75, 1.0)
        ax.set_ylim(0.75, 1.0)
        ax.set_xlabel("Nominal Coverage Level")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title(f"Reliability Diagram — {win_label}")
        ax.legend(loc="upper left", frameon=True, fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration Plot: Empirical vs Nominal Coverage by Window (EVAL Split)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — NEW: Point Forecast MAPE Degradation
# ---------------------------------------------------------------------------

def plot_fig8_point_forecast_degradation(point_table_df: pd.DataFrame, out_path: Path):
    """Figure 8: Point Forecast MAPE by Window and Model.

    Separates model-bias failure (MAPE rises post-patch) from calibration failure
    (MAPE flat but intervals under-cover).  Required diagnostic to correctly attribute
    the source of interval undercoverage.
    """
    set_plot_style()
    if point_table_df.empty:
        return

    windows    = ["post_patch (0-7d)", "post_patch (8-14d)", "post_patch (15-30d)", "baseline (>30d)"]
    win_labels = ["0-7d", "8-14d", "15-30d", "Baseline"]
    models     = sorted(point_table_df["model"].unique())
    model_colors = {"lgbm": "#007bff", "snaive": "#fd7e14"}

    fig, ax = plt.subplots(figsize=(7, 4))
    x         = np.arange(len(windows))
    bar_width = 0.35 / max(len(models), 1)

    for i, m in enumerate(models):
        m_sub = (
            point_table_df[point_table_df["model"] == m]
            .groupby("window")["mape"]
            .mean()
            .reindex(windows)
        )
        pos = x + (i - len(models) / 2.0 + 0.5) * bar_width
        ax.bar(pos, m_sub.values, bar_width,
               label=m.upper(), color=model_colors.get(m, "gray"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(win_labels)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(
        "Point Forecast MAPE by Post-Patch Window (EVAL Split)\n"
        "— Rise post-patch ⟹ model-bias failure; flat ⟹ calibration failure"
    )
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
