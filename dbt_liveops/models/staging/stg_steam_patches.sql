{{ config(
    materialized='view',
    schema='STAGING'
) }}

/*
  Staging model: stg_steam_patches
  Extracts developer announcements and patch notes from RAW.STEAM_NEWS_INGEST,
  filtering and classifying candidate game update events.
*/

WITH raw_news AS (
    SELECT
        ingest_id,
        game_id,
        app_id,
        raw_payload,
        ingested_at
    FROM {{ source('raw', 'steam_news_ingest') }}
),

-- Flatten news items from the raw JSON payload
flattened_news AS (
    SELECT
        r.game_id,
        r.app_id,
        f.value:gid::VARCHAR AS patch_id,
        f.value:title::VARCHAR AS title,
        f.value:url::VARCHAR AS url,
        f.value:feedname::VARCHAR AS feed_name,
        TO_TIMESTAMP_NTZ(f.value:date::NUMBER) AS published_at,
        DATE(TO_TIMESTAMP_NTZ(f.value:date::NUMBER)) AS patch_date
    FROM raw_news r,
    LATERAL FLATTEN(INPUT => r.raw_payload) f
    WHERE f.value:title IS NOT NULL
)

SELECT
    MD5(CONCAT(game_id, '_', COALESCE(patch_id, title), '_', TO_VARCHAR(patch_date, 'YYYY-MM-DD'))) AS patch_key,
    game_id,
    app_id,
    COALESCE(patch_id, MD5(title)) AS patch_id,
    patch_date,
    published_at,
    title,
    feed_name,
    url,
    -- Regex heuristic identifying official update notes vs community posts
    CASE 
        WHEN title RLIKE '(?i).*(update|patch|hotfix|notes|season|expansion|balance|rework).*' THEN TRUE
        ELSE FALSE
    END AS is_patch_candidate,
    -- Preliminary major release identification
    CASE
        WHEN title RLIKE '(?i).*(season|expansion|major|chapter|v[0-9]+\\.[0-9]+).*' THEN 'major'
        ELSE 'minor'
    END AS preliminary_type,
    CURRENT_TIMESTAMP() AS stg_loaded_at
FROM flattened_news
WHERE patch_date IS NOT NULL

