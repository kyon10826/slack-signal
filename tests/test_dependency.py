"""属人化マップと副担当の候補（sql/gold/04_dependency.sql）のテスト。"""

from __future__ import annotations

from datetime import date, timedelta
from importlib import resources

import duckdb

from slack_ml import pipeline
from slack_ml import warehouse as wh

LAST = date(2026, 3, 31)


def run_dependency_sql(con: duckdb.DuckDBPyConnection) -> None:
    sql = (resources.files("slack_ml") / "sql" / "gold" / "04_dependency.sql").read_text(encoding="utf-8")
    con.execute(sql)


def tiny_warehouse(messages: list[tuple], events: list[tuple]) -> duckdb.DuckDBPyConnection:
    """04_dependency.sql が読む列だけを持つ、小さな silver と gold を作る。"""
    con = duckdb.connect()
    con.execute("CREATE SCHEMA silver; CREATE SCHEMA gold;")
    con.execute(
        "CREATE TABLE silver.messages (channel_id VARCHAR, ts VARCHAR, user_id VARCHAR, "
        "is_activity BOOLEAN, is_reply BOOLEAN, thread_ts VARCHAR, posted_date DATE)"
    )
    for channel_id, ts, user_id, thread_ts, days_ago in messages:
        con.execute(
            "INSERT INTO silver.messages VALUES (?, ?, ?, TRUE, ?, ?, ?)",
            [channel_id, ts, user_id, thread_ts is not None, thread_ts, LAST - timedelta(days=days_ago)],
        )
    con.execute(
        "CREATE TABLE silver.channels AS SELECT DISTINCT channel_id, channel_id AS channel_name FROM silver.messages"
    )
    con.execute("CREATE TABLE gold.user_events (user_id VARCHAR, event_date DATE, event_type VARCHAR)")
    for user_id, days_ago in events:
        con.execute(
            "INSERT INTO gold.user_events VALUES (?, ?, 'post')", [user_id, LAST - timedelta(days=days_ago)]
        )
    return con


def thread(channel: str, n: int, asker: str, responders: list[str], days_ago: int = 1) -> list[tuple]:
    """トップレベルの投稿1件と、responders の返信を作る。"""
    ts = f"{channel}-{n}"
    rows = [(channel, ts, asker, None, days_ago)]
    rows += [(channel, f"{ts}-r{i}", r, ts, days_ago) for i, r in enumerate(responders)]
    return rows


def test_dependent_channel_needs_enough_threads_and_more_than_half_solo():
    messages = []
    for i in range(6):
        messages += thread("CA", i, "UQ", ["UK"])
    for i in range(4):
        messages += thread("CS", i, "UQ", ["UZ"])
    for i in range(6):
        messages += thread("CH", i, "UQ", ["UK" if i < 3 else "UB"])
    messages += thread("CA", 99, "UQ", ["UK"], days_ago=200)
    con = tiny_warehouse(messages, events=[("UQ", 1)])
    run_dependency_sql(con)
    rows = {
        r[0]: r[1:]
        for r in con.execute(
            "SELECT channel_id, threads, key_person_id, solo_share, is_dependent FROM gold.channel_dependency"
        ).fetchall()
    }
    assert rows["CA"] == (6, "UK", 1.0, True)
    assert rows["CS"][3] is False
    assert rows["CH"][2] == 0.5 and rows["CH"][3] is False


def test_backup_candidates_are_active_and_not_concentrated_on_one_person():
    messages = []
    for i in range(6):
        messages += thread("CB", i, "UQ", ["UK"] if i < 4 else ["UK", "UB"])
    for i in range(6):
        messages += thread("CD", i, "UP", ["UB"])
    messages += [("CB", "x1", "UR", None, 2)]
    messages += [("CB", "x2", "UI", None, 60)]
    events = [("UQ", 1), ("UK", 1), ("UB", 1), ("UR", 2), ("UI", 60)]
    con = tiny_warehouse(messages, events)
    run_dependency_sql(con)
    order = [
        r[0]
        for r in con.execute(
            "SELECT candidate_id FROM gold.backup_candidates WHERE channel_id = 'CB' ORDER BY candidate_rank"
        ).fetchall()
    ]
    assert order == ["UQ", "UR", "UB"]


def test_person_totals_add_up_across_channels():
    """人ごとの「ひとりで答えた数」は、チャンネルをまたいで足し合わせる。"""
    messages = []
    for i in range(6):
        messages += thread("CA", i, "UQ", ["UK"])
    for i in range(6):
        messages += thread("CB", i, "UQ", ["UK"] if i < 4 else ["UK", "UB"])
    for i in range(24):
        messages += thread("CW", i, "UQ", ["UH"] if i < 12 else ["UO", "UP"])
    con = tiny_warehouse(messages, events=[("UQ", 1)])
    run_dependency_sql(con)
    people = {
        r[0]: r[1:]
        for r in con.execute(
            "SELECT user_id, channels, solo, dependent_channels, heavy_channels, solo_rank "
            "FROM gold.person_dependency"
        ).fetchall()
    }
    [(total_share,)] = con.execute("SELECT SUM(solo_share) FROM gold.person_dependency").fetchall()
    [(heavy,)] = con.execute(
        "SELECT is_heavy FROM gold.channel_dependency WHERE channel_id = 'CW'"
    ).fetchall()
    assert people["UK"] == (2, 10, 2, 0, 2)
    assert people["UH"] == (1, 12, 0, 1, 1)
    assert people["UB"][1] == 0
    assert heavy is True
    assert abs(total_share - 1.0) < 1e-9


def test_candidates_already_carrying_load_go_last():
    """件数が多いチャンネルの担当者も「すでに負担がある」とみなして、候補の後ろに回す。"""
    messages = []
    for i in range(24):
        messages += thread("CW", i, "UQ", ["UH"] if i < 12 else ["UO", "UP"])
    for i in range(6):
        messages += thread("CB", i, "UQ", ["UK"] if i < 4 else ["UK", "UH"])
    messages += [("CB", "x1", "UR", None, 2)]
    con = tiny_warehouse(messages, events=[("UQ", 1), ("UH", 1), ("UR", 2)])
    run_dependency_sql(con)
    order = [
        r[0]
        for r in con.execute(
            "SELECT candidate_id FROM gold.backup_candidates WHERE channel_id = 'CB' ORDER BY candidate_rank"
        ).fetchall()
    ]
    assert order == ["UQ", "UR", "UH"]


def test_candidate_taken_by_a_heavier_channel_goes_last():
    """ひとりで答えた件数がもっと多いチャンネルで候補になっている人は、後ろに回す（同じ人に集めない）。"""
    messages = []
    for i in range(8):
        messages += thread("C1", i, "UQ1", ["UK1"] if i < 7 else ["UK1", "UX"])
    for i in range(6):
        messages += thread("C2", i, "UQ2", ["UK2"] if i < 5 else ["UK2", "UX"])
    messages += [("C2", "y1", "UY", None, 2)]
    events = [("UQ1", 1), ("UQ2", 1), ("UX", 1), ("UY", 2)]
    con = tiny_warehouse(messages, events)
    run_dependency_sql(con)

    def order(channel: str) -> list[str]:
        return [
            r[0]
            for r in con.execute(
                "SELECT candidate_id FROM gold.backup_candidates WHERE channel_id = ? ORDER BY candidate_rank",
                [channel],
            ).fetchall()
        ]

    assert order("C1") == ["UX", "UQ1"]
    assert order("C2") == ["UQ2", "UY", "UX"]


def test_pipeline_builds_the_map_from_the_test_export(export):
    """テスト用のエクスポートでは、general の U1 の投稿に答えているのは U2 だけ。"""
    cfg, _ = export
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse, read_only=True)
    [(channel, key, solo_share, dependent)] = con.execute(
        "SELECT channel_name, key_person_id, solo_share, is_dependent FROM gold.channel_dependency"
    ).fetchall()
    candidates = [
        r[0]
        for r in con.execute(
            "SELECT candidate_id FROM gold.backup_candidates ORDER BY candidate_rank"
        ).fetchall()
    ]
    con.close()
    assert (channel, key, solo_share, dependent) == ("general", "U2", 1.0, True)
    assert candidates == ["U1", "U5"]
