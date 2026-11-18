-- ==============================================================================
-- Snowflake Raw Landing Tables, Internal Stages, & File Formats
-- Database: LIVEOPS_DW | Schema: RAW
-- ==============================================================================

USE ROLE LIVEOPS_ENGINEER;
USE WAREHOUSE LIVEOPS_WH;
USE DATABASE LIVEOPS_DW;
USE SCHEMA RAW;

-- 1. Create File Formats for Data Ingestion
CREATE OR REPLACE FILE FORMAT FF_JSON
    TYPE = 'JSON'
    STRIP_OUTER_ARRAY = TRUE
    IGNORE_UTF8_ERRORS = FALSE
    COMMENT = 'Standard JSON file format for Steam Web API and SteamCharts responses';

CREATE OR REPLACE FILE FORMAT FF_PARQUET
    TYPE = 'PARQUET'
    COMPRESSION = 'SNAPPY'
    COMMENT = 'Snappy-compressed Parquet format for batch analytical files';

CREATE OR REPLACE FILE FORMAT FF_CSV
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL', 'null')
    COMMENT = 'CSV file format for daily telemetry fallbacks';

-- 2. Create Named Internal Stages for Staging Files
CREATE OR REPLACE STAGE STEAM_INGEST_STAGE
    FILE_FORMAT = FF_JSON
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Internal stage holding landing JSON dumps from SteamCharts and Steam News';

CREATE OR REPLACE STAGE PARQUET_INGEST_STAGE
    FILE_FORMAT = FF_PARQUET
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Internal stage for bulk loading Parquet datasets';

-- 3. Raw Bronze Ingestion Tables (Schema-on-Read using VARIANT)

-- 3a. SteamCharts Raw Telemetry Dump
CREATE TABLE IF NOT EXISTS STEAM_CCU_INGEST (
    ingest_id BIGINT AUTOINCREMENT PRIMARY KEY,
    game_id VARCHAR(64) NOT NULL,
    app_id NUMBER(12, 0) NOT NULL,
    raw_payload VARIANT NOT NULL,
    source_filename VARCHAR(256),
    ingested_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (game_id, DATE(ingested_at))
COMMENT = 'Raw, append-only landing table for SteamCharts hourly/daily CCU payloads';

-- 3b. Steam News API Raw Patch Announcements
CREATE TABLE IF NOT EXISTS STEAM_NEWS_INGEST (
    ingest_id BIGINT AUTOINCREMENT PRIMARY KEY,
    game_id VARCHAR(64) NOT NULL,
    app_id NUMBER(12, 0) NOT NULL,
    raw_payload VARIANT NOT NULL,
    source_filename VARCHAR(256),
    ingested_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (game_id, DATE(ingested_at))
COMMENT = 'Raw, append-only landing table for Steam News API patch announcements';

-- 3c. Raw Forecast Intervals & Residuals
CREATE TABLE IF NOT EXISTS FORECAST_INTERVALS_INGEST (
    ingest_id BIGINT AUTOINCREMENT PRIMARY KEY,
    game_id VARCHAR(64) NOT NULL,
    model VARCHAR(64) NOT NULL,
    arm VARCHAR(64) NOT NULL,
    raw_payload VARIANT NOT NULL,
    source_filename VARCHAR(256),
    ingested_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
CLUSTER BY (game_id, model, arm)
COMMENT = 'Landing table for conformal interval backtest records and residual logs';

