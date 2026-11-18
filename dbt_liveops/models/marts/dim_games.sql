{{ config(
    materialized='table',
    schema='ANALYTICS'
) }}

/*
  Dimension model: dim_games
  Stores metadata, genre attributes, and high-watermark player concurrency milestones.
*/

WITH ccu_summary AS (
    SELECT
        game_id,
        app_id,
        MIN(date) AS first_tracked_date,
        MAX(date) AS last_tracked_date,
        COUNT(DISTINCT date) AS total_tracked_days,
        MAX(y_peak) AS all_time_peak_ccu,
        ROUND(AVG(y_peak), 0) AS historical_avg_peak_ccu
    FROM {{ ref('int_daily_telemetry_cleaned') }}
    WHERE is_outage = FALSE
    GROUP BY game_id, app_id
)

SELECT
    s.game_id,
    s.app_id,
    CASE 
        WHEN s.game_id = 'apex_legends' THEN 'Apex Legends'
        WHEN s.game_id = 'path_of_exile' THEN 'Path of Exile'
        WHEN s.game_id = 'team_fortress_2' THEN 'Team Fortress 2'
        ELSE INITCAP(REPLACE(s.game_id, '_', ' '))
    END AS game_name,
    CASE
        WHEN s.game_id = 'apex_legends' THEN 'Battle Royale / Competitive Shooter'
        WHEN s.game_id = 'path_of_exile' THEN 'Action RPG / Seasonal Economy'
        WHEN s.game_id = 'team_fortress_2' THEN 'Class-Based Hero Shooter'
        ELSE 'Multiplayer Live-Service'
    END AS genre,
    s.first_tracked_date,
    s.last_tracked_date,
    s.total_tracked_days,
    s.all_time_peak_ccu,
    s.historical_avg_peak_ccu,
    CURRENT_TIMESTAMP() AS mart_created_at
FROM ccu_summary s

