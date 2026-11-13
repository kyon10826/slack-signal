"""silver 層をプロファイリングする（bronze との件数の突き合わせと、分類ごとの件数）。

使い方:
    python scripts/profile_silver.py
    python scripts/profile_silver.py configs/default.yaml
"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

QUERIES = {
    "1. bronze と silver の件数の突き合わせ（silver_rows = bronze_unique になるはず）": """
        SELECT
            (SELECT COUNT(*) FROM bronze.messages) AS bronze_rows,
            (SELECT COUNT(*) FROM (SELECT DISTINCT channel_id, ts FROM bronze.messages)) AS bronze_unique,
            (SELECT COUNT(*) FROM silver.messages) AS silver_rows
    """,
    "2. 投稿者の種類ごとの件数": """
        SELECT author_type, is_activity, COUNT(*) AS n, COUNT(DISTINCT user_id) AS users
        FROM silver.messages
        GROUP BY ALL
        ORDER BY n DESC
    """,
    "3. 活動に数える発言の内訳": """
        SELECT COALESCE(subtype, '(通常の発言)') AS subtype, is_reply, COUNT(*) AS n
        FROM silver.messages
        WHERE is_activity
        GROUP BY ALL
        ORDER BY n DESC
    """,
    "4. メンバーの返信は、誰のスレッドへの返信か": """
        SELECT COALESCE(parent_author_type, '(親が見つからない)') AS parent_author_type, COUNT(*) AS n
        FROM silver.messages
        WHERE is_activity AND is_reply
        GROUP BY ALL
        ORDER BY n DESC
    """,
    "5. リアクションは、誰の投稿に付いたか": """
        SELECT
            target_author_type,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE reactor_id = author_id) AS self_reactions
        FROM silver.reactions
        GROUP BY ALL
        ORDER BY n DESC
    """,
    "6. メンバーの一覧": """
        SELECT
            COUNT(*) AS members,
            COUNT(register_date) AS with_register_date,
            COUNT(*) FILTER (WHERE deleted) AS deleted,
            MIN(register_date) AS first_register,
            MAX(register_date) AS last_register
        FROM silver.users
    """,
    "7. 月ごとの活動（日本時間）": """
        WITH actions AS (
            SELECT user_id, posted_date AS d FROM silver.messages WHERE is_activity
            UNION ALL
            SELECT reactor_id, reacted_date_approx FROM silver.reactions
        )
        SELECT strftime(d, '%Y-%m') AS month, COUNT(DISTINCT user_id) AS active_members, COUNT(*) AS actions
        FROM actions
        GROUP BY 1
        ORDER BY 1
    """,
}

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
for title, query in QUERIES.items():
    print(f"\n== {title} ==")
    con.sql(query).show(max_rows=100)
con.close()
