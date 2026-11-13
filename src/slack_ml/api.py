"""API と Slack 連携: 今週の予測と詳しい人探しを、HTTP と Slack のスラッシュコマンドから使えるようにする。"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from collections.abc import Callable
from urllib.parse import parse_qs

import duckdb
import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request

from slack_ml import expertise
from slack_ml.config import Config, load_config
from slack_ml.scoring import reason_text

SIGNATURE_MAX_AGE_SECONDS = 60 * 5


def verify_slack_signature(
    secret: str, timestamp: str, body: bytes, signature: str, now: float | None = None
) -> bool:
    """Slack からのリクエストかを、署名で確かめる（5分より古いものは、使い回しを防ぐために断る）。"""
    try:
        sent_at = int(timestamp)
    except (TypeError, ValueError):
        return False
    if abs((now if now is not None else time.time()) - sent_at) > SIGNATURE_MAX_AGE_SECONDS:
        return False
    base = f"v0:{timestamp}:".encode() + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def post_to_slack(url: str, message: dict) -> None:
    """response_url に結果を送る。"""
    httpx.post(url, json=message, timeout=10)


class Store:
    """DWH と索引の読み出し。リクエストのたびに読み取り専用で開き、すぐに閉じる。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._index = None
        self._index_mtime = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        try:
            return duckdb.connect(str(self.cfg.paths.warehouse), read_only=True)
        except (duckdb.IOException, duckdb.ConnectionException) as e:
            raise HTTPException(
                status_code=503, detail="データを更新中です。しばらくしてから試してください"
            ) from e

    def risk(self, limit: int) -> dict:
        con = self.connect()
        try:
            tables = {
                name
                for (name,) in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'ops'"
                ).fetchall()
            }
            if "risk_scores" not in tables:
                raise HTTPException(
                    status_code=404, detail="まだ予測がありません。weekly.py を実行してください"
                )
            rows = con.execute(
                """
                WITH latest AS (
                    SELECT model, MAX(snapshot_date) AS snapshot_date
                    FROM ops.risk_scores
                    WHERE model = (
                        SELECT model FROM ops.model_evaluation WHERE selected
                        ORDER BY evaluated_at DESC LIMIT 1
                    )
                    GROUP BY model
                )
                SELECT s.snapshot_date, s.user_id, s.rank_early, s.risk, s.reasons, s.model,
                       COALESCE(NULLIF(u.display_name, ''), NULLIF(u.real_name, ''), u.name, s.user_id) AS name
                FROM ops.risk_scores AS s
                JOIN latest USING (model, snapshot_date)
                LEFT JOIN silver.users AS u USING (user_id)
                WHERE s.early
                ORDER BY s.rank_early
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            partners: dict[str, list[dict]] = {}
            if rows and "connect_suggestions" in tables:
                for user_id, partner_id, name, similarity in con.execute(
                    """
                    SELECT c.user_id, c.partner_id,
                           COALESCE(NULLIF(u.display_name, ''), NULLIF(u.real_name, ''), u.name, c.partner_id),
                           c.similarity
                    FROM ops.connect_suggestions AS c
                    LEFT JOIN silver.users AS u ON u.user_id = c.partner_id
                    WHERE c.snapshot_date = ?
                    ORDER BY c.user_id, c.partner_rank
                    """,
                    [rows[0][0]],
                ).fetchall():
                    partners.setdefault(user_id, []).append(
                        {"user_id": partner_id, "name": name, "similarity": round(similarity, 3)}
                    )
        finally:
            con.close()
        return {
            "snapshot_date": str(rows[0][0]) if rows else None,
            "model": rows[0][5] if rows else None,
            "members": [
                {
                    "rank": int(rank),
                    "user_id": user_id,
                    "name": name,
                    "risk": round(float(risk), 4),
                    "reasons": [reason_text(r) for r in json.loads(reasons)],
                    "partners": partners.get(user_id, []),
                }
                for _, user_id, rank, risk, reasons, _, name in rows
            ],
        }

    def index(self):
        """詳しい人探しの索引（weekly.py が保存したもの）。ファイルが新しくなったら読み直す。"""
        path = self.cfg.paths.artifacts_dir / "expertise.joblib"
        if not path.exists():
            raise HTTPException(status_code=404, detail="まだ索引がありません。weekly.py を実行してください")
        mtime = path.stat().st_mtime
        if self._index is None or mtime != self._index_mtime:
            self._index, self._index_mtime = expertise.load_index(path), mtime
        return self._index

    def experts(self, query: str, limit: int) -> dict:
        index = self.index()
        found = index.search(query, top_n=limit)
        con = self.connect()
        try:
            names = dict(
                con.execute(
                    "SELECT user_id, COALESCE(NULLIF(display_name, ''), NULLIF(real_name, ''), name, user_id) "
                    "FROM silver.users"
                ).fetchall()
            )
        finally:
            con.close()
        return {
            "query": query,
            "index_built_until": str(index.built_until),
            "members": [
                {
                    "user_id": user_id,
                    "name": names.get(user_id, user_id),
                    "similarity": round(similarity, 3),
                    "messages": index.message_counts[user_id],
                }
                for user_id, similarity in found
            ],
        }


def risk_message(result: dict) -> dict:
    if not result["members"]:
        return {"response_type": "ephemeral", "text": "今週の候補はいません。"}
    lines = [f"*今週の候補*（予測時点 {result['snapshot_date']}、モデル {result['model']}）"]
    for m in result["members"]:
        lines.append(f"{m['rank']}. *{m['name']}*  リスク {m['risk']:.0%}")
        lines += [f"    ・{reason}" for reason in m["reasons"]]
        if m["partners"]:
            lines.append("    つなぐ相手: " + "、".join(p["name"] for p in m["partners"]))
    return {"response_type": "ephemeral", "text": "\n".join(lines)}


def experts_message(result: dict) -> dict:
    if not result["members"]:
        return {"response_type": "ephemeral", "text": "詳しそうなメンバーが見つかりませんでした。"}
    lines = [f"*「{result['query']}」に詳しそうなメンバー*"]
    lines += [
        f"{i}. {m['name']}（話題の近さ {m['similarity']:.2f}）"
        for i, m in enumerate(result["members"], start=1)
    ]
    return {"response_type": "ephemeral", "text": "\n".join(lines)}


HELP = (
    "使い方:\n"
    "  /slackml experts 質問文  … 質問に詳しそうなメンバーを探します\n"
    "  /slackml risk           … 今週の候補と理由（CM だけ）"
)


def create_app(cfg: Config, sender: Callable[[str, dict], None] = post_to_slack) -> FastAPI:
    """API を作る。sender は Slack に結果を送る関数（テストでは差し替える）。"""
    app = FastAPI(title="slack-ml", description="メンバー活性化の予測と、詳しい人探しの API", version="0.1.0")
    store = Store(cfg)

    def require_api_key(x_api_key: str | None) -> None:
        expected = os.environ.get("SLACK_ML_API_KEY")
        if not expected:
            raise HTTPException(status_code=503, detail="API キーが設定されていません（SLACK_ML_API_KEY）")
        if not x_api_key or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status_code=401, detail="API キーが違います")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/risk")
    def risk(limit: int = Query(5, ge=1, le=50), x_api_key: str | None = Header(None)) -> dict:
        require_api_key(x_api_key)
        return store.risk(limit)

    @app.get("/v1/experts")
    def experts(
        q: str = Query(..., min_length=1, max_length=500),
        limit: int = Query(5, ge=1, le=20),
        x_api_key: str | None = Header(None),
    ) -> dict:
        require_api_key(x_api_key)
        return store.experts(q, limit)

    @app.post("/slack/commands")
    async def slack_commands(request: Request, background: BackgroundTasks) -> dict:
        secret = os.environ.get("SLACK_SIGNING_SECRET")
        if not secret:
            raise HTTPException(status_code=503, detail="Slack の Signing Secret が設定されていません")
        body = await request.body()
        if not verify_slack_signature(
            secret,
            request.headers.get("X-Slack-Request-Timestamp", ""),
            body,
            request.headers.get("X-Slack-Signature", ""),
        ):
            raise HTTPException(status_code=401, detail="Slack の署名を確かめられませんでした")

        def respond(url: str, build: Callable[[], dict]) -> None:
            """結果を作って Slack に送る。データが無いなどで作れなければ、その理由を送る。"""
            try:
                message = build()
            except HTTPException as e:
                message = {"response_type": "ephemeral", "text": str(e.detail)}
            sender(url, message)

        form = {key: values[0] for key, values in parse_qs(body.decode()).items()}
        words = form.get("text", "").strip().split(maxsplit=1)
        command = words[0].lower() if words else "help"
        response_url = form.get("response_url", "")

        if command == "experts" and len(words) == 2:
            background.add_task(respond, response_url, lambda: experts_message(store.experts(words[1], 5)))
            return {"response_type": "ephemeral", "text": "探しています。結果はこのあと届きます。"}
        if command == "risk":
            if form.get("user_id") not in cfg.slack.manager_user_ids:
                return {
                    "response_type": "ephemeral",
                    "text": "この操作はコミュニティマネージャーだけができます。",
                }
            background.add_task(respond, response_url, lambda: risk_message(store.risk(cfg.model.capacity_k)))
            return {
                "response_type": "ephemeral",
                "text": "今週の候補を集めています。結果はこのあと届きます。",
            }
        return {"response_type": "ephemeral", "text": HELP}

    return app


def app_from_env() -> FastAPI:
    """uvicorn から使う入口: SLACK_ML_CONFIG の設定ファイルで API を作る。"""
    return create_app(load_config(os.environ.get("SLACK_ML_CONFIG", "configs/local.yaml")))
