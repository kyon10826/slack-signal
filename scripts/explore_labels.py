"""離脱ラベルの定義の候補ごとに、予測時点あたりの対象者数と、陽性（離脱）の数を数える。"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

POPULATION = """
WITH data_range AS (
    SELECT MIN(event_date) AS first_date, MAX(event_date) AS last_date
    FROM gold.user_events
),
own AS (
    SELECT user_id, event_date
    FROM gold.user_events
    WHERE event_type IN ('post', 'reply', 'reaction_given')
),
snapshots AS (
    SELECT CAST(t.s AS DATE) AS snapshot_date
    FROM data_range AS r,
        generate_series(r.first_date + 56, r.last_date - 27, INTERVAL 7 DAY) AS t(s)
),
counts AS (
    SELECT
        s.snapshot_date,
        o.user_id,
        COUNT(*) FILTER (WHERE o.event_date < s.snapshot_date) AS prev_28d,
        COUNT(*) FILTER (WHERE o.event_date >= s.snapshot_date) AS next_28d,
        COUNT(*) FILTER (WHERE o.event_date >= s.snapshot_date AND o.event_date < s.snapshot_date + 14) AS next_14d
    FROM snapshots AS s
    JOIN own AS o
        ON o.event_date >= s.snapshot_date - 28
        AND o.event_date < s.snapshot_date + 28
    GROUP BY s.snapshot_date, o.user_id
)
SELECT c.*
FROM counts AS c
JOIN silver.users AS u USING (user_id)
WHERE c.prev_28d > 0
  AND u.register_date <= c.snapshot_date - 7
"""

QUERIES = {
    "1. 予測時点と対象者": f"""
        WITH population AS ({POPULATION})
        SELECT
            COUNT(DISTINCT snapshot_date) AS snapshots,
            MIN(snapshot_date) AS first_snapshot,
            MAX(snapshot_date) AS last_snapshot,
            COUNT(*) AS rows,
            COUNT(DISTINCT user_id) AS members,
            ROUND(COUNT(*) / COUNT(DISTINCT snapshot_date), 1) AS avg_population
        FROM population
    """,
    "2. ラベルの定義ごとの陽性の数": f"""
        WITH population AS ({POPULATION}),
        labeled AS (
            SELECT
                snapshot_date,
                user_id,
                next_28d = 0 AS a,
                next_14d = 0 AS b,
                next_28d <= prev_28d * 0.5 AS c
            FROM population
        ),
        long AS (
            SELECT snapshot_date, user_id, 'A: 次の28日に行動なし' AS definition, a AS positive FROM labeled
            UNION ALL
            SELECT snapshot_date, user_id, 'B: 次の14日に行動なし', b FROM labeled
            UNION ALL
            SELECT snapshot_date, user_id, 'C: 次の28日の行動が直前28日の半分以下', c FROM labeled
        )
        SELECT
            definition,
            COUNT(*) FILTER (WHERE positive) AS positives,
            ROUND(AVG(CASE WHEN positive THEN 1.0 ELSE 0.0 END), 3) AS rate,
            ROUND(COUNT(*) FILTER (WHERE positive) / COUNT(DISTINCT snapshot_date), 2) AS per_snapshot,
            COUNT(DISTINCT user_id) FILTER (WHERE positive) AS members
        FROM long
        GROUP BY definition
        ORDER BY definition
    """,
}

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
for title, query in QUERIES.items():
    print(f"\n== {title} ==")
    con.sql(query).show(max_rows=100, max_width=200)
con.close()
