from __future__ import annotations

from importlib import resources
from pathlib import Path

import duckdb

from slack_ml.ingest.slack_export import ExportTables

SCHEMAS = ("bronze", "silver", "gold", "ops")

BRONZE_DDL = {
    "users": "user_id VARCHAR, name VARCHAR, real_name VARCHAR, display_name VARCHAR, is_bot BOOLEAN, deleted BOOLEAN",
    "channels": "channel_id VARCHAR, channel_name VARCHAR, is_general BOOLEAN",
    "messages": (
        "channel_id VARCHAR, channel_name VARCHAR, ts VARCHAR, ts_epoch DOUBLE, user_id VARCHAR, "
        "subtype VARCHAR, bot_id VARCHAR, text VARCHAR, thread_ts VARCHAR, parent_user_id VARCHAR, "
        "source_file VARCHAR"
    ),
    "reactions": "channel_id VARCHAR, message_ts VARCHAR, reaction VARCHAR, user_id VARCHAR",
    "quarantine": "source_file VARCHAR, reason VARCHAR, payload VARCHAR",
}


def connect(path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB のファイルを開く"""
    path = Path(path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        for schema in SCHEMAS:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    return con


def load_bronze(con: duckdb.DuckDBPyConnection, tables: ExportTables) -> dict[str, int]:
    """取り込んだ5つの表で bronze 層を作り直し、表ごとの行数を返す。"""
    counts = {}
    for name, ddl in BRONZE_DDL.items():
        df = getattr(tables, name)
        con.execute(f"CREATE OR REPLACE TABLE bronze.{name} ({ddl})")
        if len(df):
            columns = ", ".join(col.split()[0] for col in ddl.split(", "))
            con.register("_df", df)
            con.execute(f"INSERT INTO bronze.{name} SELECT {columns} FROM _df")
            con.unregister("_df")
        counts[name] = con.execute(f"SELECT COUNT(*) FROM bronze.{name}").fetchall()[0][0]
    return counts


def run_sql_layer(con: duckdb.DuckDBPyConnection, layer: str) -> list[str]:
    """src/slack_ml/sql/<layer>/ の SQL ファイルを名前順に実行し、実行したファイル名を返す。"""
    folder = resources.files("slack_ml") / "sql" / layer
    names = sorted(p.name for p in folder.iterdir() if p.name.endswith(".sql"))
    for name in names:
        con.execute((folder / name).read_text(encoding="utf-8"))
    return names


def table_counts(con: duckdb.DuckDBPyConnection, schema: str) -> dict[str, int]:
    """スキーマの中の表ごとの行数を返す。"""
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ? ORDER BY table_name",
        [schema],
    ).fetchall()
    return {name: con.execute(f"SELECT COUNT(*) FROM {schema}.{name}").fetchall()[0][0] for (name,) in rows}
