"""gold 層をプロファイリングする（silver との件数の突き合わせと、週次 KPI の傾向）。"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

QUERIES = {
    "1. イベントの種類 × 相手の種類": """
        SELECT event_type, COALESCE(counterpart_type, '(なし)') AS counterpart_type, COUNT(*) AS n
        FROM gold.user_events
        GROUP BY ALL
        ORDER BY event_type, n DESC
    """,
    "2. silver との突き合わせ（左右の2列ずつが一致するはず）": """
        SELECT
            (SELECT COUNT(*) FROM silver.messages WHERE is_activity) AS silver_activity,
            (SELECT COUNT(*) FROM gold.user_events WHERE event_type IN ('post', 'reply')) AS gold_post_reply,
            (SELECT COUNT(*) FROM silver.reactions) AS silver_reactions,
            (SELECT COUNT(*) FROM gold.user_events WHERE event_type = 'reaction_given') AS gold_reaction_given
    """,
    "3. データの期間": """
        SELECT
            (SELECT MIN(posted_at) FROM silver.messages WHERE is_activity) AS first_activity_at,
            (SELECT MAX(posted_at) FROM silver.messages WHERE is_activity) AS last_activity_at,
            (SELECT MAX(posted_at) FROM silver.messages) AS last_message_at,
            COUNT(*) AS weeks,
            COUNT(*) FILTER (WHERE is_partial_week) AS partial_weeks
        FROM gold.weekly_kpi
    """,
    "4. 四半期ごとの週平均（7日分そろった週だけ）": """
        SELECT
            strftime(week_start, '%Y') || '-Q' || quarter(week_start) AS quarter,
            COUNT(*) AS weeks,
            ROUND(AVG(active_members), 1) AS active,
            ROUND(AVG(interacting_members), 1) AS interacting,
            ROUND(AVG(active_5plus_28d), 1) AS active_5plus,
            ROUND(AVG(posts), 1) AS posts,
            ROUND(AVG(replies), 1) AS replies,
            ROUND(AVG(reply_rate_7d), 3) AS reply_rate_7d,
            SUM(new_members) AS new_members
        FROM gold.weekly_kpi
        WHERE NOT is_partial_week
        GROUP BY 1
        ORDER BY 1
    """,
    "5. 直近8週": """
        SELECT
            week_start,
            is_partial_week AS partial,
            active_members AS active,
            interacting_members AS interacting,
            active_5plus_28d AS active_5plus,
            posts,
            replies,
            replies_to_bot AS to_bot,
            ROUND(reply_rate_7d, 3) AS reply_rate_7d,
            new_members AS new
        FROM gold.weekly_kpi
        ORDER BY week_start DESC
        LIMIT 8
    """,
    "6. 最後に行動してからの日数（データの最終日の時点）": """
        WITH last_action AS (
            SELECT user_id, MAX(event_date) AS last_date
            FROM gold.user_events
            WHERE event_type IN ('post', 'reply', 'reaction_given')
            GROUP BY user_id
        ),
        days AS (
            SELECT user_id, date_diff('day', last_date, MAX(last_date) OVER ()) AS d
            FROM last_action
        )
        SELECT
            CASE
                WHEN d <= 7 THEN '1: 7日以内'
                WHEN d <= 28 THEN '2: 8〜28日'
                WHEN d <= 90 THEN '3: 29〜90日'
                ELSE '4: 91日以上'
            END AS since_last_action,
            COUNT(*) AS members
        FROM days
        GROUP BY 1
        ORDER BY 1
    """,
}

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
for title, query in QUERIES.items():
    print(f"\n== {title} ==")
    con.sql(query).show(max_rows=100, max_width=200)
con.close()
