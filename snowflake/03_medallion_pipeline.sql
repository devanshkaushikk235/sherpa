-- ==============================================================================
-- Snowflake Medallion ELT Pipeline (Dynamic Tables Architecture)
-- Database: LIVEOPS_DW
-- ==============================================================================

USE ROLE LIVEOPS_ENGINEER;
USE WAREHOUSE LIVEOPS_WH;
USE DATABASE LIVEOPS_DW;

-- ==============================================================================
-- 1. BRONZE LAYER (RAW SCHEMA)
-- Ingestion: Ingests raw JSON data into append-only VARIANT tables.
-- ==============================================================================
USE SCHEMA RAW;

-- Ensure landing table exists
CREATE TABLE IF NOT EXISTS RAW.STEAM_CCU_INGEST (
    ingest_id BIGINT AUTOINCREMENT PRIMARY KEY,
    game_id VARCHAR(64) NOT NULL,
    app_id NUMBER(12, 0) NOT NULL,
    raw_payload VARIANT NOT NULL,
    source_filename VARCHAR(256),
    ingested_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (game_id, DATE(ingested_at));


-- ==============================================================================
-- 2. SILVER LAYER (STAGING SCHEMA)
-- Transformation: Flattens JSON arrays, casts types, flags outages, and deduplicates.
-- Implemented as Snowflake Dynamic Tables (auto-refreshed incrementally).
-- ==============================================================================
USE SCHEMA STAGING;

-- 2a. Silver Table: Cleaned Sub-Daily Samples to Daily Aggregates
CREATE OR REPLACE DYNAMIC TABLE STAGING.DT_DAILY_TELEMETRY_CLEANED
    TARGET_LAG = '1 hour'
    WAREHOUSE = LIVEOPS_WH
AS
WITH raw_data AS (
    SELECT
        game_id,
        app_id,
        raw_payload,
        ingested_at
    FROM RAW.STEAM_CCU_INGEST
),

-- Flatten the sub-daily [timestamp_ms, player_count] pairs
flattened AS (
    SELECT
        r.game_id,
        r.app_id,
        TO_TIMESTAMP_NTZ(f.value[0]::NUMBER / 1000) AS sample_time,
        f.value[1]::NUMBER AS players
    FROM raw_data r,
    LATERAL FLATTEN(INPUT => r.raw_payload) f
    WHERE f.value[0] IS NOT NULL 
      AND f.value[1] IS NOT NULL
      AND f.value[1]::NUMBER >= 0
),

daily_base AS (
    SELECT
        game_id,
        app_id,
        DATE(sample_time) AS date,
        MAX(players) AS y_peak,
        ROUND(AVG(players), 2) AS y_avg,
        COUNT(*) AS n_samples
    FROM flattened
    GROUP BY game_id, app_id, DATE(sample_time)
),

rolling_stats AS (
    SELECT
        d.*,
        -- 15-day centered rolling median for anomaly baseline
        MEDIAN(y_peak) OVER (
            PARTITION BY game_id 
            ORDER BY date 
            ROWS BETWEEN 7 PRECEDING AND 7 FOLLOWING
        ) AS rolling_median_15,
        LAG(y_peak, 7) OVER (
            PARTITION BY game_id 
            ORDER BY date
        ) AS y_peak_lag_7,
        DAYOFWEEK(date) AS day_of_week,
        CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN TRUE ELSE FALSE END AS is_weekend
    FROM daily_base d
)

SELECT
    MD5(CONCAT(game_id, '_', TO_VARCHAR(date, 'YYYY-MM-DD'))) AS ccu_key,
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
    -- Asymmetric Outage Detection: flag sudden collapses < 20% of 15d median
    CASE
        WHEN rolling_median_15 > 0 AND y_peak < (0.20 * rolling_median_15) THEN TRUE
        ELSE FALSE
    END AS is_outage,
    CURRENT_TIMESTAMP() AS silver_transformed_at
FROM rolling_stats;


-- ==============================================================================
-- 3. GOLD LAYER (ANALYTICS SCHEMA)
-- Serving: Dimension models, patch performance rollups, and capacity SLA metrics.
-- Also implemented as Dynamic Tables downstream from the Silver layer.
-- ==============================================================================
USE SCHEMA ANALYTICS;

-- 3a. Dimension Table: Games
CREATE OR REPLACE DYNAMIC TABLE ANALYTICS.DT_DIM_GAMES
    TARGET_LAG = '1 hour'
    WAREHOUSE = LIVEOPS_WH
AS
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
        WHEN game_id = 'apex_legends' THEN 'Battle Royale / Shooter'
        WHEN game_id = 'path_of_exile' THEN 'Action RPG / Seasonal'
        WHEN game_id = 'team_fortress_2' THEN 'Hero Shooter'
        ELSE 'Multiplayer LiveOps'
    END AS genre,
    MIN(date) AS first_tracked_date,
    MAX(date) AS last_tracked_date,
    COUNT(DISTINCT date) AS total_tracked_days,
    MAX(y_peak) AS all_time_peak_ccu,
    ROUND(AVG(y_peak), 0) AS historical_avg_peak_ccu,
    CURRENT_TIMESTAMP() AS gold_updated_at
FROM STAGING.DT_DAILY_TELEMETRY_CLEANED
WHERE is_outage = FALSE
GROUP BY game_id, app_id;


-- 3b. Fact Table: Daily Player Concurrency Mart
CREATE OR REPLACE DYNAMIC TABLE ANALYTICS.DT_FCT_DAILY_CONCURRENCY
    TARGET_LAG = '1 hour'
    WAREHOUSE = LIVEOPS_WH
AS
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
    -- Week-over-Week growth percentage
    ROUND((y_peak - y_peak_lag_7) / NULLIF(y_peak_lag_7, 0) * 100, 2) AS wow_growth_pct,
    CURRENT_TIMESTAMP() AS gold_updated_at
FROM STAGING.DT_DAILY_TELEMETRY_CLEANED;


-- ==============================================================================
-- 4. VERIFICATION & REFRESH COMMANDS
-- Useful commands to inspect pipeline state and trigger manual refreshes.
-- ==============================================================================

-- Manually refresh Dynamic Tables (or wait for the TARGET_LAG interval)
-- ALTER DYNAMIC TABLE STAGING.DT_DAILY_TELEMETRY_CLEANED REFRESH;
-- ALTER DYNAMIC TABLE ANALYTICS.DT_DIM_GAMES REFRESH;
-- ALTER DYNAMIC TABLE ANALYTICS.DT_FCT_DAILY_CONCURRENCY REFRESH;

-- Inspect Dynamic Table Refresh History and Lag Metrics
-- SELECT * FROM TABLE(INFORMATION_SCHEMA.DYNAMIC_TABLE_REFRESH_HISTORY())
-- ORDER BY DATA_TIMESTAMP DESC;

