{{ config(
    materialized='view',
    schema='STAGING'
) }}

/*
  Staging model: stg_steam_ccu
  Unpacks semi-structured VARIANT JSON arrays from RAW.STEAM_CCU_INGEST into
  clean, daily-aggregated player concurrency metrics (peak and average).
*/

WITH raw_telemetry AS (
    SELECT
        ingest_id,
        game_id,
        app_id,
        raw_payload,
        source_filename,
        ingested_at
    FROM {{ source('raw', 'steam_ccu_ingest') }}
),

-- Flatten the sub-daily [timestamp_ms, player_count] pairs
flattened_samples AS (
    SELECT
        r.game_id,
        r.app_id,
        -- Extract epoch milliseconds and convert to UTC timestamp
        TO_TIMESTAMP_NTZ(f.value[0]::NUMBER / 1000) AS sample_time,
        f.value[1]::NUMBER AS players
    FROM raw_telemetry r,
    LATERAL FLATTEN(INPUT => r.raw_payload) f
    WHERE f.value[0] IS NOT NULL 
      AND f.value[1] IS NOT NULL
      AND f.value[1]::NUMBER >= 0
),

-- Aggregate to calendar day
daily_aggregated AS (
    SELECT
        game_id,
        app_id,
        DATE(sample_time) AS date,
        MAX(players) AS y_peak,
        ROUND(AVG(players), 2) AS y_avg,
        COUNT(*) AS n_samples
    FROM flattened_samples
    GROUP BY game_id, app_id, DATE(sample_time)
)

SELECT
    MD5(CONCAT(game_id, '_', TO_VARCHAR(date, 'YYYY-MM-DD'))) AS ccu_key,
    game_id,
    app_id,
    date,
    y_peak,
    y_avg,
    n_samples,
    CURRENT_TIMESTAMP() AS stg_loaded_at
FROM daily_aggregated
WHERE date IS NOT NULL

