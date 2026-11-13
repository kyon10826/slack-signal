"""特徴量とラベルの表（gold.feature_snapshots）をプロファイリングする。

使い方:
    python scripts/profile_features.py
    python scripts/profile_features.py configs/default.yaml
"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

QUERIES = {
    "1. 予測時点とラベル（ラベルありの行数と陽性の数が explore_labels.py の A と一致するはず）": """
        SELECT
            COUNT(DISTINCT snapshot_date) AS snapshots,
            COUNT(*) AS rows,
            COUNT(churned) AS labeled,
            SUM(churned) AS positives,
            ROUND(AVG(churned), 3) AS rate,
            MAX(snapshot_date) FILTER (WHERE churned IS NOT NULL) AS last_labeled,
            MAX(snapshot_date) AS scoring_date,
            COUNT(*) FILTER (WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM gold.feature_snapshots))
                AS scoring_population
        FROM gold.feature_snapshots
    """,
    "2. 離脱した人としなかった人の、特徴量の平均（ratio = 離脱 ÷ 継続）": """
        WITH long AS (
            UNPIVOT (
                SELECT * EXCLUDE (snapshot_date, user_id)
                FROM gold.feature_snapshots
                WHERE churned IS NOT NULL
            )
            ON COLUMNS(* EXCLUDE (churned))
            INTO NAME feature VALUE value
        )
        SELECT
            feature,
            ROUND(AVG(value) FILTER (WHERE churned = 0), 2) AS stayed,
            ROUND(AVG(value) FILTER (WHERE churned = 1), 2) AS churned,
            ROUND(AVG(value) FILTER (WHERE churned = 1) / NULLIF(AVG(value) FILTER (WHERE churned = 0), 0), 2)
                AS ratio
        FROM long
        GROUP BY feature
        ORDER BY ratio
    """,
    "3. 同じ人の連続した陽性と、その後に戻ってきた割合": """
        WITH flags AS (
            SELECT
                p.snapshot_date,
                p.user_id,
                EXISTS (
                    SELECT 1
                    FROM gold.feature_snapshots AS q
                    WHERE q.user_id = p.user_id
                        AND q.churned = 1
                        AND q.snapshot_date = p.snapshot_date - 7
                ) AS continues_previous,
                p.snapshot_date + 28 + 56 <= (SELECT MAX(snapshot_date) FROM gold.feature_snapshots)
                    AS can_observe_return,
                EXISTS (
                    SELECT 1
                    FROM gold.user_events AS e
                    WHERE e.user_id = p.user_id
                        AND e.event_type IN ('post', 'reply', 'reaction_given')
                        AND e.event_date >= p.snapshot_date + 28
                        AND e.event_date < p.snapshot_date + 28 + 56
                ) AS returned_within_56d
            FROM gold.feature_snapshots AS p
            WHERE p.churned = 1
        )
        SELECT
            COUNT(*) AS positives,
            COUNT(*) FILTER (WHERE NOT continues_previous) AS episodes,
            COUNT(*) FILTER (WHERE can_observe_return) AS observable,
            COUNT(*) FILTER (WHERE can_observe_return AND returned_within_56d) AS returned,
            ROUND(AVG(CASE WHEN returned_within_56d THEN 1.0 ELSE 0.0 END) FILTER (WHERE can_observe_return), 3)
                AS return_rate
        FROM flags
    """,
}

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
for title, query in QUERIES.items():
    print(f"\n== {title} ==")
    con.sql(query).show(max_rows=100, max_width=200)
con.close()
