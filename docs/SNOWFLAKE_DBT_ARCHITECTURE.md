# Enterprise LiveOps Data Platform: Snowflake & dbt Architecture

## 1. System Overview

This architecture transforms the project into a modern, enterprise-grade Cloud Data Warehouse platform built on **Snowflake** and **dbt (data build tool)**. It applies the **Medallion Architecture** pattern to live-service video game telemetry (*Apex Legends*, *Path of Exile*, *Team Fortress 2*):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               1. BRONZE (RAW SCHEMA)                                   │
│  - Unmodified, append-only landing tables storing raw JSON in Snowflake VARIANT        │
│  - Tables: RAW.STEAM_CCU_INGEST, RAW.STEAM_NEWS_INGEST                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ dbt Stage View / Unpack & Flatten
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              2. SILVER (STAGING SCHEMA)                                │
│  - Flattened, deduplicated, UTC-standardized daily CCU aggregates                      │
│  - Automated asymmetric outage detection (< 20% 15-day rolling median)                 │
│  - Models: stg_steam_ccu, stg_steam_patches, int_daily_telemetry_cleaned,              │
│            int_patch_surge_windows                                                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ dbt Dimensional Modeling & Marts
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             3. GOLD (ANALYTICS SCHEMA)                                 │
│  - Star-schema dimensional marts serving LiveOps capacity planning & ML inference      │
│  - Models: dim_games, fct_daily_player_concurrency, fct_patch_performance,             │
│            fct_liveops_capacity_risk                                                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
forecast-reliability/
├── snowflake/
│   ├── 01_setup_roles_warehouses.sql   # DDL: Roles, X-Small Warehouse, DB, & Schemas
│   └── 02_raw_tables_and_stages.sql    # DDL: File Formats, Internal Stages, VARIANT tables
├── dbt_liveops/
│   ├── dbt_project.yml                 # dbt project definition & schema routing
│   ├── profiles.yml.example            # Snowflake profile connection template
│   ├── models/
│   │   ├── staging/                    # Views unpacking VARIANT JSON arrays
│   │   │   ├── _staging_models.yml
│   │   │   ├── stg_steam_ccu.sql
│   │   │   └── stg_steam_patches.sql
│   │   ├── intermediate/               # Outage detection & [-7d, +7d] patch windows
│   │   │   ├── _intermediate_models.yml
│   │   │   ├── int_daily_telemetry_cleaned.sql
│   │   │   └── int_patch_surge_windows.sql
│   │   └── marts/                      # Star-schema analytics fact & dim tables
│   │       ├── _marts_models.yml
│   │       ├── dim_games.sql
│   │       ├── fct_daily_player_concurrency.sql
│   │       ├── fct_patch_performance.sql
│   │       └── fct_liveops_capacity_risk.sql
│   └── tests/                          # Singular data contract assertions
│       ├── assert_peak_gte_average.sql
│       └── assert_no_future_dates.sql
├── src/ingest/
│   └── snowflake_loader.py             # Python ELT loader (JSON -> Snowflake VARIANT)
└── .sqlfluff                           # Snowflake SQL linting and styling configuration
```

---

## 3. Setup & Quickstart Guide

### Step 1: Provision Snowflake Infrastructure
Open your Snowflake console and execute:
1. `snowflake/01_setup_roles_warehouses.sql` (Creates `LIVEOPS_WH`, `LIVEOPS_DW`, and roles).
2. `snowflake/02_raw_tables_and_stages.sql` (Creates stages, file formats, and `VARIANT` tables).

### Step 2: Configure Environment Variables
Add your Snowflake credentials to your `.env` file:
```bash
SNOWFLAKE_ACCOUNT="your-account-identifier"
SNOWFLAKE_USER="your-username"
SNOWFLAKE_PASSWORD="your-password"
SNOWFLAKE_ROLE="LIVEOPS_ENGINEER"
SNOWFLAKE_WAREHOUSE="LIVEOPS_WH"
SNOWFLAKE_DATABASE="LIVEOPS_DW"
SNOWFLAKE_SCHEMA="RAW"
```

Configure your `~/.dbt/profiles.yml`:
```bash
# Copy example configuration
cp dbt_liveops/profiles.yml.example ~/.dbt/profiles.yml
```

### Step 3: Ingest Raw Telemetry into Snowflake Bronze Layer
Run the Snowflake ingestion command:
```bash
python run.py snowflake-load
```
This loads raw `steamcharts_chart.json` and `steam_news.json` dumps directly into `RAW.STEAM_CCU_INGEST` and `RAW.STEAM_NEWS_INGEST` as semi-structured `VARIANT` objects.

### Step 4: Run dbt Transformations & Data Tests
Execute the dbt pipeline to compile and materialize all staging, intermediate, and gold models:
```bash
# Test connection
dbt debug --project-dir dbt_liveops

# Run transformations
dbt run --project-dir dbt_liveops

# Run automated data quality contracts & tests
dbt test --project-dir dbt_liveops
```

### Step 5: Lint SQL with SQLFluff
Ensure code quality adheres to production Snowflake standards:
```bash
sqlfluff lint dbt_liveops/models/
```

---

## 4. Engineering Trade-offs & Production Design Decisions

Key architectural decisions implemented in this platform:

1. **Semi-Structured Ingestion with `VARIANT`**:
   - Telemetry from Steam arrives as nested JSON arrays (`[[ts, players], ...]`). Rather than running fragile in-memory Python parsers, raw payloads are landed immutably into Snowflake `VARIANT` columns.
   - Transformations are pushed down into Snowflake using `LATERAL FLATTEN()` and vectorized date conversions.

2. **Cost Optimization & Compute Efficiency**:
   - The virtual warehouse `LIVEOPS_WH` is configured as `XSMALL` with `AUTO_SUSPEND = 60` seconds. Compute drops to 0 cost immediately after dbt jobs finish.
   - Clustering keys are defined on `(game_id, date)` to prune partition scans during multi-year time-series lookups.

3. **Data Quality Circuit Breakers & Invariants**:
   - **Schema contracts**: Primary key uniqueness, null checks, and accepted enum values on patch types (`['major', 'minor']`).
   - **Singular business tests**: `assert_peak_gte_average.sql` asserts that peak CCU is strictly greater than or equal to average CCU across all daily rows.
   - **Outage sanitization**: Asymmetric outage filter identifies telemetry dropouts without discarding genuine patch surge spikes.

4. **Time Travel for Reproducibility**:
   - `LIVEOPS_DW` is configured with a 7-day retention period, enabling instant time travel queries (`AT (OFFSET => ...)`) to audit exact historical data states prior to patch deployments.

