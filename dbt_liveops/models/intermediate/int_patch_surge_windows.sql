{{ config(
    materialized='table',
    schema='STAGING'
) }}

/*
  Intermediate model: int_patch_surge_windows
  Joins verified patches with daily telemetry across a [-7, +7] day temporal window.
  Computes pre-patch baseline, launch day peak, 7-day post-patch average,
  empirical shock magnitude M, and flags patch overlaps.
*/

WITH patches AS (
    SELECT
        patch_key,
        game_id,
        app_id,
        patch_id,
        patch_date,
        title,
        preliminary_type
    FROM {{ ref('stg_steam_patches') }}
    WHERE is_patch_candidate = TRUE
),

telemetry AS (
    SELECT
        game_id,
        date,
        y_peak
    FROM {{ ref('int_daily_telemetry_cleaned') }}
    WHERE is_outage = FALSE -- Exclude corrupted outage points from surge metrics
),

-- Window join [-7, +7] days relative to patch launch date
patch_windows AS (
    SELECT
        p.patch_key,
        p.game_id,
        p.app_id,
        p.patch_id,
        p.patch_date,
        p.title,
        p.preliminary_type,
        t.date AS obs_date,
        DATEDIFF('day', p.patch_date, t.date) AS days_since_patch,
        t.y_peak
    FROM patches p
    JOIN telemetry t
      ON t.game_id = p.game_id
     AND t.date BETWEEN DATEADD('day', -7, p.patch_date) AND DATEADD('day', 7, p.patch_date)
),

patch_aggregates AS (
    SELECT
        patch_key,
        game_id,
        app_id,
        patch_id,
        patch_date,
        title,
        preliminary_type,
        AVG(CASE WHEN days_since_patch BETWEEN -7 AND -1 THEN y_peak END) AS pre_patch_avg_peak,
        MAX(CASE WHEN days_since_patch = 0 THEN y_peak END) AS patch_day_peak,
        AVG(CASE WHEN days_since_patch BETWEEN 1 AND 7 THEN y_peak END) AS post_patch_avg_peak
    FROM patch_windows
    GROUP BY patch_key, game_id, app_id, patch_id, patch_date, title, preliminary_type
),

classified_patches AS (
    SELECT
        a.*,
        -- Launch day surge %
        ROUND((patch_day_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100, 2) AS launch_surge_pct,
        -- Week 1 sustained surge %
        ROUND((post_patch_avg_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0) * 100, 2) AS week1_sustained_surge_pct,
        -- Empirical shock magnitude M
        ROUND((post_patch_avg_peak - pre_patch_avg_peak) / NULLIF(pre_patch_avg_peak, 0), 4) AS empirical_magnitude_m,
        -- Check distance to previous patch for overlap
        LAG(patch_date) OVER (PARTITION BY game_id ORDER BY patch_date) AS prev_patch_date
    FROM patch_aggregates a
)

SELECT
    patch_key,
    game_id,
    app_id,
    patch_id,
    patch_date,
    title,
    pre_patch_avg_peak,
    patch_day_peak,
    post_patch_avg_peak,
    launch_surge_pct,
    week1_sustained_surge_pct,
    empirical_magnitude_m,
    -- Overlap flag: previous patch occurred within 14 days
    CASE 
        WHEN prev_patch_date IS NOT NULL AND DATEDIFF('day', prev_patch_date, patch_date) <= 14 THEN TRUE
        ELSE FALSE
    END AS overlaps_prev_patch,
    -- Final classification: Major if |M| >= 0.20 or tagged as preliminary major
    CASE
        WHEN ABS(empirical_magnitude_m) >= 0.20 OR preliminary_type = 'major' THEN 'major'
        ELSE 'minor'
    END AS final_patch_type,
    CURRENT_TIMESTAMP() AS int_created_at
FROM classified_patches

