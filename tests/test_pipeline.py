"""パイプライン全体のテスト。小さなテスト用のエクスポートで、取り込みから gold まで通して確かめる。"""

from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta, timezone

import duckdb
import pytest

from slack_ml import pipeline, quality
from slack_ml import warehouse as wh

JST = timezone(timedelta(hours=9))


def run(cfg) -> dict:
    return pipeline.run(cfg, log=lambda *_: None)


def query(cfg, sql: str, params: list | None = None) -> list[tuple]:
    con = duckdb.connect(str(cfg.paths.warehouse), read_only=True)
    try:
        return con.execute(sql, params or []).fetchall()
    finally:
        con.close()


def test_pipeline_passes_all_error_checks(export):
    cfg, expected = export
    result = run(cfg)
    assert result["bronze"]["messages"] == expected["bronze_messages"]
    failed = [r["name"] for r in result["quality"] if r["severity"] == "error" and not r["passed"]]
    assert failed == []


def test_rerun_gives_same_tables(export):
    """冪等性: 2回実行しても、すべての表の行数が同じ。"""
    cfg, _ = export
    run(cfg)
    con = wh.connect(cfg.paths.warehouse, read_only=True)
    first = {schema: wh.table_counts(con, schema) for schema in ("bronze", "silver", "gold")}
    con.close()
    run(cfg)
    con = wh.connect(cfg.paths.warehouse, read_only=True)
    second = {schema: wh.table_counts(con, schema) for schema in ("bronze", "silver", "gold")}
    con.close()
    assert first == second


def test_duplicates_are_removed(export):
    """2つのバッチに入った同じメッセージは、silver で1件になる。リアクションも1件になる。"""
    cfg, expected = export
    run(cfg)
    [(bronze, silver)] = query(
        cfg, "SELECT (SELECT COUNT(*) FROM bronze.messages), (SELECT COUNT(*) FROM silver.messages)"
    )
    assert bronze == expected["bronze_messages"]
    assert silver == expected["bronze_messages"] - 1
    [(reactions,)] = query(cfg, "SELECT COUNT(*) FROM silver.reactions")
    assert reactions == expected["reactions"]


def test_author_types(export):
    """投稿者の分類ごとの件数が、テスト用データを作ったときの件数と一致する。"""
    cfg, expected = export
    run(cfg)
    counts = dict(query(cfg, "SELECT author_type, COUNT(*) FROM silver.messages GROUP BY 1"))
    assert counts == {
        "member": expected["member_activity"] + expected["member_system"],
        "member_app": expected["member_app"],
        "bot": expected["bot"],
        "no_user": expected["no_user"],
    }
    [(activity,)] = query(cfg, "SELECT COUNT(*) FROM silver.messages WHERE is_activity")
    assert activity == expected["member_activity"]


def test_dates_are_jst_not_file_date(export):
    """日本時間の朝の投稿は、太平洋時間では前の日のファイルに入るが、posted_date は日本時間の日付になる。"""
    cfg, _ = export
    run(cfg)
    [(posted_date, source_file)] = query(
        cfg,
        "SELECT posted_date, source_file FROM silver.messages WHERE user_id = 'U1' AND is_activity ORDER BY ts LIMIT 1",
    )
    assert posted_date == date(2026, 1, 5)
    assert source_file.endswith("2026-01-04.json")


def test_reply_to_bot_is_not_member_interaction(export):
    """日報ボットへのコメントは、相手が bot の返信になり、誰かが「返信をもらった」ことにはならない。"""
    cfg, _ = export
    run(cfg)
    counts = dict(
        query(
            cfg,
            "SELECT counterpart_type, COUNT(*) FROM gold.user_events WHERE user_id = 'U2' AND event_type = 'reply' GROUP BY 1",
        )
    )
    assert counts["bot"] > 0
    [(received_from_bot,)] = query(
        cfg,
        "SELECT COUNT(*) FROM gold.user_events WHERE event_type = 'reply_received' AND counterpart_id = 'UBOT'",
    )
    assert received_from_bot == 0


def test_self_reply_is_not_member_interaction(export):
    """自分のスレッドへの返信は、相手が self になり、「返信をもらった」には数えない。"""
    cfg, _ = export
    run(cfg)
    counts = dict(
        query(
            cfg,
            "SELECT counterpart_type, COUNT(*) FROM gold.user_events "
            "WHERE user_id = 'U1' AND event_type = 'reply' GROUP BY 1",
        )
    )
    assert counts.get("self", 0) > 0
    [(from_self,)] = query(
        cfg,
        "SELECT COUNT(*) FROM gold.user_events WHERE event_type = 'reply_received' AND counterpart_id = user_id",
    )
    assert from_self == 0


def test_churn_label(export):
    """来なくなった U4 は離脱（1）、毎日来る U1 は継続（0）になる。"""
    cfg, _ = export
    run(cfg)
    labels = dict(
        query(
            cfg,
            "SELECT user_id, MAX(churned) FROM gold.feature_snapshots WHERE churned IS NOT NULL GROUP BY 1",
        )
    )
    assert labels["U4"] == 1
    assert labels["U1"] == 0


def test_quality_check_catches_duplicates(export):
    """silver に重複を入れると、error のチェックが失敗する（チェックが本当に効くことの確認）。"""
    cfg, _ = export
    run(cfg)
    con = wh.connect(cfg.paths.warehouse)
    con.execute("INSERT INTO silver.messages SELECT * FROM silver.messages LIMIT 1")
    results = {r["name"]: r for r in quality.run_checks(con, "test")}
    con.close()
    assert not results["silver_messages_unique"]["passed"]
    assert not results["silver_matches_bronze"]["passed"]


def test_features_do_not_use_future_data(export, tmp_path):
    """リークの検査: 予測時点 t より後のデータを消して作り直しても、t の特徴量は変わらない。"""
    cfg, _ = export
    run(cfg)
    snapshots = [
        row[0] for row in query(cfg, "SELECT DISTINCT snapshot_date FROM gold.feature_snapshots ORDER BY 1")
    ]
    t = snapshots[len(snapshots) // 2]
    columns = [row[0] for row in query(cfg, "DESCRIBE gold.feature_snapshots") if row[0] != "churned"]
    select = (
        f"SELECT {', '.join(columns)} FROM gold.feature_snapshots WHERE snapshot_date = ? ORDER BY user_id"
    )
    full = query(cfg, select, [t])
    assert full, "テスト用データに、予測時点 t の対象者がいない"

    truncated_path = tmp_path / "truncated.duckdb"
    shutil.copy(cfg.paths.warehouse, truncated_path)
    t_epoch = datetime(t.year, t.month, t.day, tzinfo=JST).timestamp()
    con = wh.connect(truncated_path)
    con.execute("DELETE FROM bronze.messages WHERE ts_epoch >= ?", [t_epoch])
    con.execute(
        "DELETE FROM bronze.reactions AS r WHERE NOT EXISTS "
        "(SELECT 1 FROM bronze.messages AS m WHERE m.channel_id = r.channel_id AND m.ts = r.message_ts)"
    )
    wh.run_sql_layer(con, "silver")
    wh.run_sql_layer(con, "gold")
    truncated = con.execute(select, [t]).fetchall()
    [(labels,)] = con.execute(
        "SELECT COUNT(churned) FROM gold.feature_snapshots WHERE snapshot_date = ?", [t]
    ).fetchall()
    con.close()

    assert truncated == full
    assert labels == 0


@pytest.mark.parametrize("bad_key", ["warehous", "export"])
def test_config_rejects_unknown_keys(tmp_path, bad_key):
    """設定ファイルの書き間違いは、実行してすぐにエラーになる。"""
    from slack_ml.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(f"paths:\n  {bad_key}: x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_config_reads_users_to_exclude_from_scoring(tmp_path):
    """予測の対象から外す人は、設定ファイルのリストから読む。"""
    from slack_ml.config import load_config

    path = tmp_path / "local.yaml"
    path.write_text("model:\n  exclude_from_scoring: [U1, U2]\n", encoding="utf-8")
    assert load_config(path).model.exclude_from_scoring == ("U1", "U2")
    assert load_config(None).model.exclude_from_scoring == ()


def test_config_tells_how_to_fix_removed_sections(tmp_path):
    """やめた機能（効果検証）の設定が残っていたら、消すように伝える。"""
    from slack_ml.config import load_config

    path = tmp_path / "old.yaml"
    path.write_text("experiment:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="experiment: の節を消してください"):
        load_config(path)
