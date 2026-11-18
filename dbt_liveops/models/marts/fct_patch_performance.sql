{{ config(
    materialized='table',
    schema='ANALYTICS'
) }}

/*
  Fact model: fct_patch_performance
  Aggregates launch surges, week-1 decay profiles, and volatility by game and patch type.
  Replaces EA LiveOps Query 1 & Query 5 with an automated production dimensional mart.
*/

WITH surges AS (
    SELECT
        game_id,
        final_patch_type,
        patch_key,
        patch_day_peak,
        pre_patch_avg_peak,
        post_patch_avg_peak,
        launch_surge_pct,
        week1_sustained_surge_pct,
        empirical_magnitude_m
    FROM {{ ref('int_patch_surge_windows') }}
    WHERE pre_patch_avg_peak > 0
)

SELECT
    MD5(CONCAT(game_id, '_', final_patch_type)) AS performance_key,
    game_id,
    final_patch_type,
    COUNT(DISTINCT patch_key) AS total_patches_evaluated,
    ROUND(AVG(launch_surge_pct), 1) AS avg_launch_surge_pct,
    ROUND(MEDIAN(launch_surge_pct), 1) AS median_launch_surge_pct,
    ROUND(AVG(week1_sustained_surge_pct), 1) AS avg_week1_sustained_surge_pct,
    ROUND(AVG(empirical_magnitude_m), 4) AS avg_magnitude_m,
    ROUND(STDDEV(launch_surge_pct), 1) AS surge_volatility_stddev,
    CURRENT_TIMESTAMP() AS mart_created_at
FROM surges
GROUP BY game_id, final_patch_type

