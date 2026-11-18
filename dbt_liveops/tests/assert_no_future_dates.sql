-- Singular Test: assert_no_future_dates
-- Fails if any telemetry timestamp is recorded in the future relative to UTC execution time.

SELECT
    ccu_key,
    game_id,
    date
FROM {{ ref('stg_steam_ccu') }}
WHERE date > CURRENT_DATE()

