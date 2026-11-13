"""bronze 層をプロファイリングする（件数や分布を見て、取り込みが正しいかを確かめる）。

使い方:
    python scripts/profile_bronze.py
    python scripts/profile_bronze.py configs/default.yaml
"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

QUERIES = {
    "1. 表と行数": """
        SELECT 'users' AS table_name, COUNT(*) AS n FROM bronze.users
        UNION ALL SELECT 'channels', COUNT(*) FROM bronze.channels
        UNION ALL SELECT 'messages', COUNT(*) FROM bronze.messages
        UNION ALL SELECT 'reactions', COUNT(*) FROM bronze.reactions
        UNION ALL SELECT 'quarantine', COUNT(*) FROM bronze.quarantine
    """,
    "2. inspect_export.py と突き合わせる数字": """
        SELECT
            COUNT(*) AS messages,
            COUNT(*) FILTER (WHERE thread_ts IS NOT NULL AND thread_ts <> ts) AS replies,
            (SELECT COUNT(DISTINCT channel_id || '/' || message_ts) FROM bronze.reactions)
                AS messages_with_reactions,
            COUNT(DISTINCT user_id) AS posters
        FROM bronze.messages
    """,
    "3. subtype ごとの件数": """
        SELECT COALESCE(subtype, '(通常の発言)') AS subtype, COUNT(*) AS n
        FROM bronze.messages
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 10
    """,
    "4. 月別の件数（ファイル名の日付）と、JST の日付とずれる件数": r"""
        SELECT
            left(file_date, 7) AS month,
            COUNT(*) AS n,
            COUNT(*) FILTER (WHERE jst_date <> file_date) AS date_differs
        FROM (
            SELECT
                regexp_extract(source_file, '(\d{4}-\d{2}-\d{2})\.json$', 1) AS file_date,
                strftime(to_timestamp(ts_epoch) AT TIME ZONE 'Asia/Tokyo', '%Y-%m-%d') AS jst_date
            FROM bronze.messages
        )
        GROUP BY 1
        ORDER BY 1
    """,
    "5. 1メッセージに付いたリアクションの人数の分布": """
        SELECT n_reactions, COUNT(*) AS messages
        FROM (
            SELECT m.channel_id, m.ts, COUNT(r.user_id) AS n_reactions
            FROM bronze.messages AS m
            LEFT JOIN bronze.reactions AS r
                ON r.channel_id = m.channel_id AND r.message_ts = m.ts
            GROUP BY m.channel_id, m.ts
        )
        GROUP BY 1
        ORDER BY 1
        LIMIT 15
    """,
    "6. 返信かどうか × bot_id の有無 × users.json のボット判定": """
        SELECT
            m.thread_ts IS NOT NULL AND m.thread_ts <> m.ts AS is_reply,
            m.bot_id IS NOT NULL AS has_bot_id,
            COALESCE(u.is_bot, false) AS user_is_bot,
            COUNT(*) AS n,
            COUNT(DISTINCT m.user_id) AS users,
            COUNT(DISTINCT m.bot_id) AS bot_ids
        FROM bronze.messages AS m
        LEFT JOIN bronze.users AS u ON u.user_id = m.user_id
        GROUP BY ALL
        ORDER BY n DESC
    """,
    "7. bot_id ごとの件数（上位10）": """
        SELECT
            bot_id,
            COUNT(*) AS n,
            COUNT(DISTINCT user_id) AS users,
            COUNT(DISTINCT channel_id) AS channels,
            COUNT(*) FILTER (WHERE thread_ts IS NOT NULL AND thread_ts <> ts) AS replies,
            COUNT(*) FILTER (WHERE subtype IS NULL) AS no_subtype
        FROM bronze.messages
        WHERE bot_id IS NOT NULL
        GROUP BY bot_id
        ORDER BY n DESC
        LIMIT 10
    """,
    "8. ファイル名の日付は、どのタイムゾーンの日付と一致するか": r"""
        SELECT
            thread_ts IS NOT NULL AND thread_ts <> ts AS is_reply,
            tz,
            COUNT(*) AS n,
            COUNT(*) FILTER (
                WHERE strftime(to_timestamp(ts_epoch) AT TIME ZONE tz, '%Y-%m-%d') = file_date
            ) AS matches
        FROM (
            SELECT
                ts,
                thread_ts,
                ts_epoch,
                regexp_extract(source_file, '(\d{4}-\d{2}-\d{2})\.json$', 1) AS file_date
            FROM bronze.messages
        )
        CROSS JOIN (VALUES ('Asia/Tokyo'), ('UTC'), ('America/Los_Angeles'), ('America/New_York')) AS t(tz)
        GROUP BY ALL
        ORDER BY is_reply, matches DESC
    """,
    "9. ボット関係の投稿者（人間の名前は伏せる）": """
        SELECT
            m.bot_id,
            CASE
                WHEN m.user_id IS NULL THEN '(user なし)'
                WHEN u.is_bot THEN COALESCE(u.name, m.user_id)
                ELSE '(人間のメンバー)'
            END AS poster,
            COALESCE(m.subtype, '(なし)') AS subtype,
            COUNT(*) AS n,
            MIN(strftime(to_timestamp(m.ts_epoch) AT TIME ZONE 'Asia/Tokyo', '%Y-%m')) AS first_month,
            MAX(strftime(to_timestamp(m.ts_epoch) AT TIME ZONE 'Asia/Tokyo', '%Y-%m')) AS last_month
        FROM bronze.messages AS m
        LEFT JOIN bronze.users AS u ON u.user_id = m.user_id
        WHERE m.bot_id IS NOT NULL OR u.is_bot OR m.user_id IS NULL
        GROUP BY ALL
        ORDER BY n DESC
        LIMIT 20
    """,
    "10. 投稿の最も多い bot_id と、人間のやり取り": """
        WITH top_bot AS (
            SELECT bot_id
            FROM bronze.messages
            WHERE bot_id IS NOT NULL
            GROUP BY bot_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
        ),
        human_threads AS (
            SELECT DISTINCT m.channel_id, m.thread_ts
            FROM bronze.messages AS m
            LEFT JOIN bronze.users AS u ON u.user_id = m.user_id
            WHERE m.thread_ts <> m.ts
              AND m.bot_id IS NULL
              AND NOT COALESCE(u.is_bot, false)
        )
        SELECT
            b.thread_ts IS NOT NULL AND b.thread_ts <> b.ts AS is_reply,
            COUNT(*) AS n,
            COUNT(h.thread_ts) AS got_human_reply,
            COUNT(*) FILTER (WHERE NOT pu.is_bot) AS parent_is_human
        FROM bronze.messages AS b
        LEFT JOIN human_threads AS h ON h.channel_id = b.channel_id AND h.thread_ts = b.ts
        LEFT JOIN bronze.users AS pu ON pu.user_id = b.parent_user_id
        WHERE b.bot_id = (SELECT bot_id FROM top_bot)
        GROUP BY ALL
        ORDER BY is_reply
    """,
}

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
for title, query in QUERIES.items():
    print(f"\n== {title} ==")
    con.sql(query).show(max_rows=100)
con.close()
