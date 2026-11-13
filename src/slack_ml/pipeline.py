from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

from slack_ml import quality
from slack_ml import warehouse as wh
from slack_ml.config import Config
from slack_ml.ingest.slack_export import read_export


class QualityError(RuntimeError):
    """error の品質チェックが1つ以上失敗した。"""


def run(cfg: Config, log=print) -> dict:
    """パイプラインを最初から最後まで実行し、各段階の結果をまとめて返す。"""
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now()
    seconds: dict[str, float] = {}
    status = "failed"
    con = wh.connect(cfg.paths.warehouse)
    try:
        log(f"実行ID: {run_id}")

        t = time.perf_counter()
        metadata = cfg.paths.metadata_export if cfg.paths.metadata_export.exists() else None
        tables = read_export(cfg.paths.export_dir, metadata=metadata, exclude_users=cfg.privacy.exclude_users)
        bronze = wh.load_bronze(con, tables)
        seconds["bronze"] = time.perf_counter() - t
        log(f"[bronze]  {seconds['bronze']:5.1f}秒  messages {bronze['messages']:,} 行")

        for layer in ("silver", "gold"):
            t = time.perf_counter()
            wh.run_sql_layer(con, layer)
            seconds[layer] = time.perf_counter() - t
            log(f"[{layer:<7}] {seconds[layer]:5.1f}秒")

        t = time.perf_counter()
        results = quality.run_checks(con, run_id)
        seconds["quality"] = time.perf_counter() - t
        log(f"[quality] {seconds['quality']:5.1f}秒")
        for r in results:
            mark = "OK  " if r["passed"] else ("NG  " if r["severity"] == "error" else "警告")
            log(f"  {mark} {r['name']:<32} 違反 {r['violations']:>6}  {r['description']}")

        failed = [r["name"] for r in results if r["severity"] == "error" and not r["passed"]]
        if failed:
            raise QualityError(f"error の品質チェックが失敗しました: {', '.join(failed)}")

        status = "success"
        return {"run_id": run_id, "bronze": bronze, "quality": results, "seconds": seconds}
    finally:
        con.execute(
            "CREATE TABLE IF NOT EXISTS ops.pipeline_runs ("
            "run_id VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, seconds VARCHAR)"
        )
        con.execute(
            "INSERT INTO ops.pipeline_runs VALUES (?, ?, ?, ?, ?)",
            [run_id, started_at, datetime.now(), status, json.dumps(seconds)],
        )
        con.close()
