-- Singular Test: assert_peak_gte_average
-- Fails if any daily observation records a peak concurrent player count lower than the daily average.

SELECT
    ccu_key,
    game_id,
    date,
    y_peak,
    y_avg
FROM {{ ref('stg_steam_ccu') }}
WHERE y_peak < y_avg

