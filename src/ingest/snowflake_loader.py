"""Production Snowflake ELT Ingestion Module.
Loads raw Steam telemetry (JSON/CSV) and analytical artifacts (Parquet)
into Snowflake Bronze landing tables (RAW.STEAM_CCU_INGEST, etc.) using VARIANT columns.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_snowflake_connection():
    """Builds and returns a Snowflake connection using environment variables."""
    try:
        import snowflake.connector
    except ImportError:
        raise ImportError(
            "snowflake-connector-python is not installed. "
            "Install it via: pip install snowflake-connector-python[pandas]"
        )

    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    role = os.getenv("SNOWFLAKE_ROLE", "LIVEOPS_ENGINEER")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "LIVEOPS_WH")
    database = os.getenv("SNOWFLAKE_DATABASE", "LIVEOPS_DW")
    schema = os.getenv("SNOWFLAKE_SCHEMA", "RAW")

    if not all([account, user]):
        raise ValueError(
            "Missing required Snowflake environment variables: SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER. "
            "Please configure your .env file."
        )

    conn_params: Dict[str, Any] = {
        "account": account,
        "user": user,
        "role": role,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
    }

    if password:
        conn_params["password"] = password
    else:
        # Fall back to browser SSO or external authenticator if no password provided
        conn_params["authenticator"] = os.getenv("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")

    logger.info("Connecting to Snowflake account=%s, user=%s, db=%s, schema=%s", account, user, database, schema)
    return snowflake.connector.connect(**conn_params)


ALLOWED_TABLES = frozenset({"STEAM_CCU_INGEST", "STEAM_NEWS_INGEST", "FORECAST_INTERVALS_INGEST"})


def ingest_raw_json(
    game_id: str,
    app_id: int,
    json_path: Path | str,
    target_table: str = "STEAM_CCU_INGEST",
) -> int:
    """Ingests a raw JSON file (SteamCharts or Steam News) into a Snowflake VARIANT table.
    
    Args:
        game_id: Unique game identifier string (e.g. 'apex_legends')
        app_id: Steam Application ID
        json_path: Path to the raw JSON file
        target_table: Name of the raw landing table in LIVEOPS_DW.RAW
        
    Returns:
        Number of records loaded.
    """
    if target_table not in ALLOWED_TABLES:
        raise ValueError(f"Invalid target table '{target_table}'. Allowed: {sorted(ALLOWED_TABLES)}")

    path = Path(json_path)
    if not path.exists():
        logger.warning("File not found: %s. Skipping ingestion.", path)
        return 0

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Wrap raw list or dict into iterable records
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # If payload has an outer container (like Steam News {'appnews': {'newsitems': [...]}}), preserve it
        records = [data]
    else:
        records = [{"raw_value": data}]

    conn = get_snowflake_connection()
    cursor = conn.cursor()

    try:
        query = f"""
            INSERT INTO RAW.{target_table} (game_id, app_id, raw_payload, source_filename)
            SELECT %s, %s, PARSE_JSON(%s), %s
        """
        rows = [
            (game_id, app_id, json.dumps(rec), path.name)
            for rec in records
        ]

        # Execute in batches of 1000 to optimize network roundtrips
        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            cursor.executemany(query, batch)
            total_inserted += len(batch)

        conn.commit()
        logger.info(
            "Successfully loaded %d records from %s into RAW.%s for %s",
            total_inserted, path.name, target_table, game_id
        )
        return total_inserted
    finally:
        cursor.close()
        conn.close()


def ingest_all_raw_data(data_root: Path | str) -> Dict[str, int]:
    """Scans data/raw/ directory and loads all available Steam JSON files into Snowflake.
    
    Args:
        data_root: Root data folder containing data/raw/<game_id>/...
        
    Returns:
        Dictionary mapping table names to total ingested records.
    """
    root = Path(data_root) / "raw"
    if not root.exists():
        logger.error("Raw directory %s does not exist.", root)
        return {}

    app_id_map = {
        "apex_legends": 1172470,
        "path_of_exile": 238960,
        "team_fortress_2": 440,
    }

    counts = {"STEAM_CCU_INGEST": 0, "STEAM_NEWS_INGEST": 0}

    for game_dir in root.iterdir():
        if not game_dir.is_dir():
            continue
        game_id = game_dir.name
        app_id = app_id_map.get(game_id, 0)

        # 1. CCU Chart JSON
        chart_file = game_dir / "steamcharts_chart.json"
        if chart_file.exists():
            n = ingest_raw_json(game_id, app_id, chart_file, "STEAM_CCU_INGEST")
            counts["STEAM_CCU_INGEST"] += n

        # 2. Steam News JSON
        news_file = game_dir / "steam_news.json"
        if news_file.exists():
            n = ingest_raw_json(game_id, app_id, news_file, "STEAM_NEWS_INGEST")
            counts["STEAM_NEWS_INGEST"] += n

    logger.info("Snowflake ingestion summary: %s", counts)
    return counts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Snowflake Raw Ingestion CLI")
    parser.add_argument("--data-root", default="data", help="Root data folder")
    parser.add_argument("--game", default=None, help="Filter by specific game ID")
    args = parser.parse_args()

    ingest_all_raw_data(args.data_root)

