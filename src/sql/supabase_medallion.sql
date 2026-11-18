-- ==============================================================================
-- Supabase (PostgreSQL) Medallion Architecture: Silver & Gold Analytical Marts
-- Run this in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query -> Run)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. SILVER LAYER: Cleaned Telemetry with Rolling Baseline & Outage Flags
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.silver_cleaned_telemetry AS
WITH base AS (
    SELECT
        game_id,
        app_id,
        date,
        y_peak,
        y_avg,
        is_outage,
        is_imputed
    FROM public.liveops_series
),
rolling_metrics AS (
    SELECT
        b.*,
        -- 15-day centered rolling average baseline (PostgreSQL window function)
        ROUND(AVG(y_peak) OVER (
            PARTITION BY game_id
            ORDER BY date
            ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING
        )::NUMERIC, 2) AS rolling_baseline_15,
        LAG(y_peak, 7) OVER (
            PARTITION BY game_id
            ORDER BY date
        ) AS y_peak_lag_7,
        EXTRACT(DOW FROM date) AS day_of_week,
        CASE WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
    FROM base b
)
SELECT
    MD5(CONCAT(game_id, '_', date::TEXT)) AS ccu_key,
    game_id,
    app_id,
    date,
    y_peak,
    y_avg,
    rolling_baseline_15,
    y_peak_lag_7,
    day_of_week,
    is_weekend,
    is_outage,
    is_imputed
FROM rolling_metrics;


-- ------------------------------------------------------------------------------
-- 2. GOLD LAYER: Dimension Table (dim_games)
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gold_dim_games AS
SELECT
    game_id,
    app_id,
    CASE 
        WHEN game_id = 'apex_legends' THEN 'Apex Legends'
        WHEN game_id = 'path_of_exile' THEN 'Path of Exile'
        WHEN game_id = 'team_fortress_2' THEN 'Team Fortress 2'
        ELSE INITCAP(REPLACE(game_id, '_', ' '))
    END AS game_name,
    CASE
        WHEN game_id = 'apex_legends' THEN 'Battle Royale / Competitive Shooter'
        WHEN game_id = 'path_of_exile' THEN 'Action RPG / Seasonal Economy'
        WHEN game_id = 'team_fortress_2' THEN 'Class-Based Hero Shooter'
        ELSE 'Multiplayer Live-Service'
    END AS genre,
    MIN(date) AS first_tracked_date,
    MAX(date) AS last_tracked_date,
    COUNT(DISTINCT date) AS total_tracked_days,
    MAX(y_peak) AS all_time_peak_ccu,
    ROUND(AVG(y_peak)::NUMERIC, 0) AS historical_avg_peak_ccu
FROM public.liveops_series
WHERE is_outage = FALSE
GROUP BY game_id, app_id;


-- ------------------------------------------------------------------------------
-- 3. GOLD LAYER: Fact Table (fct_daily_player_concurrency)
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gold_fct_daily_player_concurrency AS
WITH daily AS (
    SELECT * FROM public.silver_cleaned_telemetry
),
patches AS (
    SELECT
        game_id,
        date AS patch_date,
        patch_id,
        title,
        patch_type
    FROM public.liveops_patches
),
daily_with_last_patch AS (
    SELECT
        d.*,
        p.patch_date AS last_patch_date,
        p.patch_id AS last_patch_id,
        p.patch_type AS last_patch_type,
        (d.date - p.patch_date) AS days_since_last_patch,
        ROW_NUMBER() OVER (
            PARTITION BY d.game_id, d.date 
            ORDER BY p.patch_date DESC
        ) AS rn
    FROM daily d
    LEFT JOIN patches p
      ON p.game_id = d.game_id
     AND p.patch_date <= d.date
)
SELECT
    ccu_key,
    game_id,
    app_id,
    date,
    y_peak,
    y_avg,
    rolling_baseline_15,
    y_peak_lag_7,
    day_of_week,
    is_weekend,
    is_outage,
    last_patch_date,
    last_patch_id,
    last_patch_type,
    COALESCE(days_since_last_patch, 999) AS days_since_last_patch,
    CASE
        WHEN days_since_last_patch BETWEEN 0 AND 7 THEN 'post_patch_0_7d'
        WHEN days_since_last_patch BETWEEN 8 AND 14 THEN 'post_patch_8_14d'
        WHEN days_since_last_patch BETWEEN 15 AND 30 THEN 'post_patch_15_30d'
        ELSE 'baseline_gt_30d'
    END AS post_patch_stratum,
    ROUND(((y_peak - y_peak_lag_7) / NULLIF(y_peak_lag_7, 0) * 100)::NUMERIC, 2) AS wow_growth_pct
FROM daily_with_last_patch
WHERE rn = 1;


-- ------------------------------------------------------------------------------
-- 4. GOLD LAYER: Patch Performance & Surge Summary Mart
-- ------------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.gold_fct_patch_performance AS
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
    WHERE s.is_outage = FALSE
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
    ROUND(AVG((patch_day_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100)::NUMERIC, 1) AS avg_launch_surge_pct,
    ROUND(AVG((post_patch_avg_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100)::NUMERIC, 1) AS avg_week1_sustained_surge_pct
FROM patch_aggregates
WHERE pre_patch_avg_peak > 0
GROUP BY game_id, patch_type;
