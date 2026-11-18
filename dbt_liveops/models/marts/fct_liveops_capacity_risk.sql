{{ config(
    materialized='table',
    schema='ANALYTICS'
) }}

/*
  Fact model: fct_liveops_capacity_risk
  Simulates asymmetric cloud compute capacity risk across varying penalty ratios r.
  Under-provisioning (y > hi) penalty: r * (y - hi)
  Over-provisioning (hi > y) idle penalty: 1.0 * (hi - y)
*/

WITH games AS (
    SELECT DISTINCT game_id FROM {{ ref('dim_games') }}
),

penalties AS (
    SELECT 2.0 AS penalty_ratio, 'Low Asymmetry (2x)' AS scenario_label
    UNION ALL
    SELECT 5.0, 'Moderate Asymmetry (5x)'
    UNION ALL
    SELECT 10.0, 'Critical SLA Asymmetry (10x)'
    UNION ALL
    SELECT 20.0, 'Catastrophic Outage Penalty (20x)'
),

game_concurrency AS (
    SELECT
        game_id,
        post_patch_stratum,
        AVG(y_peak) AS mean_y_peak,
        STDDEV(y_peak) AS stddev_y_peak,
        COUNT(*) AS total_days
    FROM {{ ref('fct_daily_player_concurrency') }}
    WHERE is_outage = FALSE
    GROUP BY game_id, post_patch_stratum
)

SELECT
    MD5(CONCAT(g.game_id, '_', g.post_patch_stratum, '_', TO_VARCHAR(p.penalty_ratio))) AS risk_key,
    g.game_id,
    g.post_patch_stratum,
    p.penalty_ratio,
    p.scenario_label,
    g.total_days,
    ROUND(g.mean_y_peak, 0) AS avg_concurrency,
    -- Estimated normalized financial risk index based on stratum volatility and under-provisioning penalty
    ROUND(
        (p.penalty_ratio * (g.stddev_y_peak * 0.10) + 1.0 * (g.stddev_y_peak * 0.90)) / NULLIF(g.mean_y_peak, 0),
        4
    ) AS estimated_risk_index,
    CURRENT_TIMESTAMP() AS mart_created_at
FROM game_concurrency g
CROSS JOIN penalties p

