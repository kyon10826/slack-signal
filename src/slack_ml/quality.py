from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb


@dataclass(frozen=True)
class Check:
    name: str
    severity: str
    description: str
    sql: str


CHECKS = [
    Check(
        "bronze_quarantine_empty",
        "warn",
        "取り込みで隔離したレコードが無い",
        "SELECT COUNT(*) FROM bronze.quarantine",
    ),
    Check(
        "silver_messages_unique",
        "error",
        "silver.messages は (channel_id, ts) で一意",
        "SELECT COUNT(*) FROM (SELECT channel_id, ts FROM silver.messages GROUP BY ALL HAVING COUNT(*) > 1)",
    ),
    Check(
        "silver_matches_bronze",
        "error",
        "silver.messages の行数 = 重複を除いた bronze.messages の行数（結合で行が増えていない）",
        """
        SELECT abs(
            (SELECT COUNT(*) FROM silver.messages)
            - (SELECT COUNT(*) FROM (SELECT DISTINCT channel_id, ts FROM bronze.messages))
        )
        """,
    ),
    Check(
        "silver_users_unique",
        "error",
        "silver.users は user_id で一意",
        "SELECT COUNT(*) FROM (SELECT user_id FROM silver.users GROUP BY ALL HAVING COUNT(*) > 1)",
    ),
    Check(
        "silver_unknown_authors",
        "warn",
        "users に載っていない投稿者がいない",
        "SELECT COUNT(*) FROM silver.messages WHERE author_type = 'unknown'",
    ),
    Check(
        "silver_reply_parent_found",
        "warn",
        "メンバーの返信は、すべて親のメッセージが見つかる",
        "SELECT COUNT(*) FROM silver.messages WHERE is_activity AND is_reply AND parent_author_type IS NULL",
    ),
    Check(
        "gold_events_match_messages",
        "error",
        "イベントログの投稿と返信の数 = silver の活動に数える発言の数",
        """
        SELECT abs(
            (SELECT COUNT(*) FROM silver.messages WHERE is_activity)
            - (SELECT COUNT(*) FROM gold.user_events WHERE event_type IN ('post', 'reply'))
        )
        """,
    ),
    Check(
        "gold_events_match_reactions",
        "error",
        "イベントログの reaction_given の数 = silver.reactions の行数",
        """
        SELECT abs(
            (SELECT COUNT(*) FROM silver.reactions)
            - (SELECT COUNT(*) FROM gold.user_events WHERE event_type = 'reaction_given')
        )
        """,
    ),
    Check(
        "gold_features_unique",
        "error",
        "gold.feature_snapshots は (snapshot_date, user_id) で一意",
        """
        SELECT COUNT(*)
        FROM (SELECT snapshot_date, user_id FROM gold.feature_snapshots GROUP BY ALL HAVING COUNT(*) > 1)
        """,
    ),
    Check(
        "gold_features_in_population",
        "error",
        "予測の対象者は、直近28日に行動し、参加から7日以上たっている",
        "SELECT COUNT(*) FROM gold.feature_snapshots WHERE days_since_last_action > 28 OR tenure_days < 7",
    ),
    Check(
        "gold_labels_only_when_observed",
        "error",
        "ラベルが NULL なのは、次の28日を観測しきれていない予測時点だけ",
        """
        SELECT COUNT(*)
        FROM gold.feature_snapshots
        WHERE (churned IS NULL) <> (snapshot_date + 28 > (SELECT MAX(snapshot_date) FROM gold.feature_snapshots))
        """,
    ),
    Check(
        "gold_label_rate_in_range",
        "warn",
        "離脱の割合が 1%〜50% の範囲にある（極端なら、ラベルか対象者の定義を疑う）",
        "SELECT CASE WHEN AVG(churned) BETWEEN 0.01 AND 0.5 THEN 0 ELSE 1 END FROM gold.feature_snapshots",
    ),
    Check(
        "gold_kpi_weeks_contiguous",
        "error",
        "週次 KPI の週が抜けなく並んでいる",
        """
        SELECT COALESCE(date_diff('day', MIN(week_start), MAX(week_start)) // 7 + 1 - COUNT(*), 0)
        FROM gold.weekly_kpi
        """,
    ),
    Check(
        "gold_kpi_members_bounded",
        "error",
        "交流したメンバー数 <= 活動したメンバー数 <= メンバー数",
        """
        SELECT COUNT(*)
        FROM gold.weekly_kpi
        WHERE interacting_members > active_members
            OR active_members > (SELECT COUNT(*) FROM silver.users)
        """,
    ),
    Check(
        "gold_dependency_shares_valid",
        "error",
        "属人化マップの割合は 0〜1 で、ひとりで答えた数は答えた数を超えない",
        """
        SELECT COUNT(*)
        FROM gold.channel_dependency
        WHERE key_share NOT BETWEEN 0 AND 1
            OR solo_share NOT BETWEEN 0 AND 1
            OR key_solo > key_answered
        """,
    ),
    Check(
        "gold_backup_not_key_person",
        "error",
        "副担当の候補に、担当者本人が入っていない",
        "SELECT COUNT(*) FROM gold.backup_candidates WHERE candidate_id = key_person_id",
    ),
    Check(
        "gold_person_solo_matches_threads",
        "error",
        "人ごとの「ひとりで答えた数」の合計 = 答えた人が1人だけのスレッドの数",
        """
        SELECT abs(
            (SELECT COALESCE(SUM(solo), 0) FROM gold.person_dependency)
            - (
                SELECT COUNT(*)
                FROM (SELECT channel_id, thread_ts FROM gold.thread_responses GROUP BY ALL HAVING COUNT(*) = 1)
            )
        )
        """,
    ),
]


def run_checks(con: duckdb.DuckDBPyConnection, run_id: str) -> list[dict]:
    """すべてのチェックを実行し、結果を ops.quality_results に追記して返す。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.quality_results ("
        "run_id VARCHAR, checked_at TIMESTAMP, name VARCHAR, severity VARCHAR, violations BIGINT, passed BOOLEAN)"
    )
    checked_at = datetime.now()
    results = []
    for check in CHECKS:
        violations = con.execute(check.sql).fetchall()[0][0]
        passed = violations == 0
        con.execute(
            "INSERT INTO ops.quality_results VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, checked_at, check.name, check.severity, violations, passed],
        )
        results.append(
            {
                "name": check.name,
                "severity": check.severity,
                "violations": violations,
                "passed": passed,
                "description": check.description,
            }
        )
    return results
