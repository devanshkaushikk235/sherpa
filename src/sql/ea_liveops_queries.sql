-- ==============================================================================
-- EA LiveOps & Infrastructure Analytics - 6 Strategic SQL Queries
-- Run these in Supabase SQL Editor on your synced tables.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Query 1: Post-Patch Concurrency Surge & 7-Day Decay Rate
-- Business Question: How much does concurrency surge on patch day versus pre-patch
-- baseline, and what is the 7-day retention profile across major vs minor updates?
-- ------------------------------------------------------------------------------
WITH patch_windows AS (
    SELECT 
        s.game_id,
        p.patch_id,
        p.patch_type,
        p.date AS patch_date,
        s.date AS obs_date,
        (s.date - p.date) AS days_since_patch,
        s.y_peak
    FROM public.liveops_patches p
    JOIN public.liveops_series s 
      ON s.game_id = p.game_id 
     AND s.date BETWEEN p.date - INTERVAL '7 days' AND p.date + INTERVAL '7 days'
),
patch_aggregates AS (
    SELECT
        game_id,
        patch_id,
        patch_type,
        AVG(CASE WHEN days_since_patch BETWEEN -7 AND -1 THEN y_peak END) AS pre_patch_avg_peak,
        MAX(CASE WHEN days_since_patch = 0 THEN y_peak END) AS patch_day_peak,
        AVG(CASE WHEN days_since_patch BETWEEN 1 AND 7 THEN y_peak END) AS post_patch_avg_peak
    FROM patch_windows
    GROUP BY game_id, patch_id, patch_type
)
SELECT 
    game_id,
    patch_type,
    COUNT(*) AS total_patches,
    ROUND(AVG((patch_day_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100)::numeric, 1) AS avg_launch_surge_pct,
    ROUND(AVG((post_patch_avg_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100)::numeric, 1) AS avg_week1_sustained_surge_pct
FROM patch_aggregates
WHERE pre_patch_avg_peak > 0
GROUP BY game_id, patch_type
ORDER BY game_id, patch_type;


-- ------------------------------------------------------------------------------
-- Query 2: Week-1 Under-Provisioning Shortfall Rate (Apex Legends Headline)
-- Business Question: How often did actual player count break through the 90% forecast
-- upper bound (hi) during the first 7 days following a patch for each recalibration arm?
-- ------------------------------------------------------------------------------
SELECT 
    arm,
    COUNT(*) AS total_evaluations,
    COUNT(CASE WHEN y_true > hi THEN 1 END) AS underprovisioned_events,
    ROUND(COUNT(CASE WHEN y_true > hi THEN 1 END)::numeric / COUNT(*) * 100, 2) AS shortfall_rate_pct,
    ROUND(AVG(hi - lo)::numeric, 0) AS mean_interval_width,
    ROUND(AVG(CASE WHEN y_true > hi THEN (y_true - hi) ELSE 0 END)::numeric, 0) AS avg_capacity_deficit_players
FROM public.liveops_intervals
WHERE game_id = 'apex_legends'
  AND level = 0.9
  AND in_patch_window = TRUE
  AND days_since_patch BETWEEN 0 AND 7
GROUP BY arm
ORDER BY shortfall_rate_pct ASC;


-- ------------------------------------------------------------------------------
-- Query 3: Empirical Coverage Recovery Across Post-Patch Strata
-- Business Question: How fast does each arm restore nominal 90% coverage as time
-- elapses from the patch date (0-7d vs 8-14d vs 15-30d vs baseline)?
-- ------------------------------------------------------------------------------
SELECT 
    arm,
    ROUND(MAX(CASE WHEN "window" = 'baseline (>30d)' THEN coverage END)::numeric * 100, 1) AS cov_baseline,
    ROUND(MAX(CASE WHEN "window" = 'post_patch (0-7d)' THEN coverage END)::numeric * 100, 1) AS cov_0_7d,
    ROUND(MAX(CASE WHEN "window" = 'post_patch (8-14d)' THEN coverage END)::numeric * 100, 1) AS cov_8_14d,
    ROUND(MAX(CASE WHEN "window" = 'post_patch (15-30d)' THEN coverage END)::numeric * 100, 1) AS cov_15_30d,
    ROUND(MAX(CASE WHEN "window" = 'post_patch (0-7d)' THEN mean_width END)::numeric, 0) AS width_0_7d
FROM public.liveops_metrics
WHERE model = 'lgbm' 
  AND level = 0.9
GROUP BY arm
ORDER BY cov_0_7d DESC;


-- ------------------------------------------------------------------------------
-- Query 4: Top 5 Worst Patch Shocks Ranked by Infrastructure Deficit
-- Business Question: Which historical patches generated the largest cumulative
-- player concurrency overruns above our fixed baseline forecast?
-- ------------------------------------------------------------------------------
SELECT 
    p.game_id,
    p.date AS patch_date,
    p.title AS patch_title,
    p.patch_type,
    ROUND(SUM(GREATEST(0, i.y_true - i.hi))::numeric, 0) AS cumulative_unprovisioned_player_hours,
    MAX(i.y_true) AS peak_actual_ccu,
    MAX(i.hi) AS max_forecasted_upper_bound
FROM public.liveops_patches p
JOIN public.liveops_intervals i
  ON i.game_id = p.game_id
 AND i.arm = 'fixed'
 AND i.level = 0.9
 AND i.target_date BETWEEN p.date AND p.date + INTERVAL '7 days'
GROUP BY p.game_id, p.date, p.title, p.patch_type
ORDER BY cumulative_unprovisioned_player_hours DESC
LIMIT 5;


-- ------------------------------------------------------------------------------
-- Query 5: Operational Cost Frontier Under Asymmetric Loss (c_under = 5x c_over)
-- Business Question: What is the optimal recalibration policy when under-provisioning
-- penalties are 5x more costly than maintaining idle cloud capacity?
-- ------------------------------------------------------------------------------
SELECT 
    c.arm,
    ROUND(AVG(CASE WHEN c.penalty_ratio = 2.0 THEN c.norm_cost END)::numeric, 3) AS cost_ratio_2x,
    ROUND(AVG(CASE WHEN c.penalty_ratio = 5.0 THEN c.norm_cost END)::numeric, 3) AS cost_ratio_5x,
    ROUND(AVG(CASE WHEN c.penalty_ratio = 10.0 THEN c.norm_cost END)::numeric, 3) AS cost_ratio_10x,
    ROUND(AVG(m.mean_width)::numeric, 0) AS avg_interval_width
FROM public.liveops_capacity_sensitivity c
JOIN public.liveops_metrics m
  ON m.game_id = c.game_id 
 AND m.model = c.model 
 AND m.arm = c.arm 
 AND m.level = c.level
WHERE c.model = 'lgbm' 
  AND c.level = 0.9
  AND m."window" = 'post_patch (0-7d)'
GROUP BY c.arm
ORDER BY cost_ratio_5x ASC;


-- ------------------------------------------------------------------------------
-- Query 6: Weekly Rolling Coverage Degradation (Window Function)
-- Business Question: Using a 7-day rolling window function over target dates,
-- track the exact date when coverage drops below 85% after an Apex Legends patch.
-- ------------------------------------------------------------------------------
WITH daily_eval AS (
    SELECT 
        target_date,
        arm,
        COUNT(*) AS total_points,
        SUM(CASE WHEN covered THEN 1 ELSE 0 END) AS covered_points
    FROM public.liveops_intervals
    WHERE game_id = 'apex_legends'
      AND level = 0.9
    GROUP BY target_date, arm
)
SELECT 
    target_date,
    arm,
    ROUND(
        AVG(covered_points::numeric / NULLIF(total_points, 0)) 
        OVER (PARTITION BY arm ORDER BY target_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) * 100, 
        1
    ) AS rolling_7d_coverage_pct
FROM daily_eval
ORDER BY target_date DESC, arm;
