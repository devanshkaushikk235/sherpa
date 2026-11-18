-- ==============================================================================
-- Snowflake Infrastructure Setup: Roles, Warehouses, Databases & Schemas
-- Run this in Snowflake Worksheets using ACCOUNTADMIN or SYSADMIN.
-- ==============================================================================

-- 1. Create Dedicated Role for LiveOps Data Engineering
USE ROLE USERADMIN;
CREATE ROLE IF NOT EXISTS LIVEOPS_ENGINEER;

-- 2. Grant Role Hierarchy
USE ROLE SECURITYADMIN;
GRANT ROLE LIVEOPS_ENGINEER TO ROLE SYSADMIN;

-- 3. Create Compute Virtual Warehouse with Auto-Suspend & Auto-Resume
USE ROLE SYSADMIN;
CREATE WAREHOUSE IF NOT EXISTS LIVEOPS_WH
    WITH 
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60          -- Suspend after 60 seconds of inactivity to minimize credits
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Virtual warehouse for LiveOps telemetry ingestion, dbt transforms, and capacity modeling';

-- 4. Create LiveOps Database
CREATE DATABASE IF NOT EXISTS LIVEOPS_DW
    DATA_RETENTION_TIME_IN_DAYS = 7 -- Time Travel retention for disaster recovery and point-in-time backtesting
    COMMENT = 'LiveOps Cloud Server Capacity & CCU Analytics Data Warehouse';

-- 5. Create Medallion Schemas inside LIVEOPS_DW
USE DATABASE LIVEOPS_DW;

-- RAW (Bronze): Unmodified append-only landing zone storing raw JSON/payloads in VARIANT
CREATE SCHEMA IF NOT EXISTS RAW
    COMMENT = 'Bronze Layer: Raw semi-structured telemetry dumps from Steam APIs';

-- STAGING (Silver Pre-transformed): Cleaned views and conformed entities
CREATE SCHEMA IF NOT EXISTS STAGING
    COMMENT = 'Silver Layer: Cleaned, deduplicated, and typed telemetry streams';

-- ANALYTICS (Gold): Business marts, dimensional models, and capacity SLA analytics
CREATE SCHEMA IF NOT EXISTS ANALYTICS
    COMMENT = 'Gold Layer: High-performance dimensional models for LiveOps, BI, and ML inference';

-- 6. Grant Permissions to LIVEOPS_ENGINEER
GRANT USAGE ON WAREHOUSE LIVEOPS_WH TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON ALL SCHEMAS IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON FUTURE SCHEMAS IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON ALL TABLES IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON ALL VIEWS IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;
GRANT ALL PRIVILEGES ON FUTURE VIEWS IN DATABASE LIVEOPS_DW TO ROLE LIVEOPS_ENGINEER;

