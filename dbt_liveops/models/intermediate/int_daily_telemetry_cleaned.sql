{{ config(
    materialized='table',
    schema='STAGING'
) }}

/*
  Intermediate model: int_daily_telemetry_cleaned
  Cleans the daily CCU series:
  - Calculates 15-day rolling median
  - Flags asymmetric outages (y_peak < 0.20 * median) while preserving organic surge spikes
  - Adds temporal features (day of week, weekend indicator, 7-day lags)
*/

WITH base AS (
    SELECT
        ccu_key,
        game_id,
        app_id,
        date,
        y_peak,
        y_avg,
        n_samples
    FROM {{ ref('stg_steam_ccu') }}
),

rolling_stats AS (
    SELECT
        b.*,
        -- 15-day centered window median (7 preceding, current, 7 following)
        MEDIAN(y_peak) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING
        ) AS rolling_median_15,
        -- 7-day lag for seasonal comparison
        LAG(y_peak, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS y_peak_lag_7,
        DAYOFWEEK(date) AS day_of_week,
        CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
    FROM base b
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
    n_samples,
    -- Outage detection: sharp asymmetric collapses < 20% of local median
    CASE
        WHEN rolling_median_15 > 0 AND y_peak < (0.20 * rolling_median_15) THEN TRUE
        ELSE FALSE
    END AS is_outage,
    CURRENT_TIMESTAMP() AS int_created_at
FROM rolling_stats

