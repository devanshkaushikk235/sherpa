{{ config(
    materialized='table',
    schema='ANALYTICS'
) }}

/*
  Fact model: fct_daily_player_concurrency
  Daily player counts enriched with patch proximity, post-patch window strata,
  and moving-average baselines.
*/

WITH daily AS (
    SELECT
        ccu_key,
        game_id,
        app_id,
        date,
        y_peak,
        y_avg,
        rolling_median_15,
        y_peak_lag_7,
        day_of_week,
        is_weekend,
        is_outage
    FROM {{ ref('int_daily_telemetry_cleaned') }}
),

patches AS (
    SELECT
        game_id,
        patch_date,
        patch_id,
        title,
        final_patch_type
    FROM {{ ref('int_patch_surge_windows') }}
),

-- Find the latest patch occurring on or before each daily observation
daily_with_last_patch AS (
    SELECT
        d.*,
        p.patch_date AS last_patch_date,
        p.patch_id AS last_patch_id,
        p.final_patch_type AS last_patch_type,
        DATEDIFF('day', p.patch_date, d.date) AS days_since_last_patch,
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
    rolling_median_15,
    y_peak_lag_7,
    day_of_week,
    is_weekend,
    is_outage,
    last_patch_date,
    last_patch_id,
    last_patch_type,
    COALESCE(days_since_last_patch, 999) AS days_since_last_patch,
    -- Stratum classification matching LiveOps operational SLAs
    CASE
        WHEN days_since_last_patch BETWEEN 0 AND 7 THEN 'post_patch_0_7d'
        WHEN days_since_last_patch BETWEEN 8 AND 14 THEN 'post_patch_8_14d'
        WHEN days_since_last_patch BETWEEN 15 AND 30 THEN 'post_patch_15_30d'
        ELSE 'baseline_gt_30d'
    END AS post_patch_stratum,
    -- Week-over-Week percentage growth
    ROUND((y_peak - y_peak_lag_7) / NULLIF(y_peak_lag_7, 0) * 100, 2) AS wow_growth_pct,
    CURRENT_TIMESTAMP() AS mart_created_at
FROM daily_with_last_patch
WHERE rn = 1

