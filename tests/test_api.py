"""API と Slack 連携のテスト。テスト用のエクスポートで週次の処理を通してから、API を呼ぶ。"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import time
from datetime import timedelta
from urllib.parse import urlencode

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from slack_ml import api, churn, expertise, pipeline, scoring
from slack_ml import warehouse as wh
from slack_ml.config import SlackConfig

API_KEY = "test-api-key"
SECRET = "test-signing-secret"
MANAGER = "UCM"


@pytest.fixture
def service(export, monkeypatch):
    """週次の処理（予測、索引、つなぐ相手）を済ませた DWH と、API のクライアント。"""
    cfg, _ = export
    cfg = dataclasses.replace(cfg, slack=SlackConfig(manager_user_ids=(MANAGER,)))
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse)
    labeled = churn.load_labeled(con)
    scored = scoring.score_latest(labeled, scoring.load_latest(con), "logistic")
    scoring.save_scores(con, scored)
    summary = pd.DataFrame(
        {
            "population": ["early"],
            "model": ["logistic"],
            "rows": [10],
            "positives": [1],
            "pr_auc": [0.5],
            "precision@5": [0.2],
            "recall@5": [1.0],
            "lift": [2.0],
        }
    )
    scoring.save_evaluation(con, summary, "logistic", 5)
    messages = expertise.load_messages(con)
    con.close()
    messages["clean"] = (
        messages["user_id"]
        .map({"U1": "pandas と numpy の集計", "U2": "pandas の groupby と numpy", "U3": "Figma の配色"})
        .fillna("その他の話題")
        + " "
        + messages["clean"]
    )
    index = expertise.build_index(messages, messages["posted_date"].max() + timedelta(days=1), min_messages=1)
    expertise.save_index(index, cfg.paths.artifacts_dir / "expertise.joblib")

    sent = []
    monkeypatch.setenv("SLACK_ML_API_KEY", API_KEY)
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SECRET)
    client = TestClient(api.create_app(cfg, sender=lambda url, message: sent.append((url, message))))
    return client, sent


def slack_request(
    client, text: str, user_id: str = MANAGER, timestamp: int | None = None, secret: str = SECRET
):
    body = urlencode(
        {"command": "/slackml", "text": text, "user_id": user_id, "response_url": "https://hooks.example/r"}
    ).encode()
    ts = str(timestamp if timestamp is not None else int(time.time()))
    signature = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:".encode() + body, hashlib.sha256).hexdigest()
    return client.post(
        "/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": ts,
            "X-Slack-Signature": signature,
        },
    )


def test_health(service):
    client, _ = service
    assert client.get("/health").json() == {"status": "ok"}


def test_risk_requires_the_api_key(service):
    client, _ = service
    assert client.get("/v1/risk").status_code == 401
    assert client.get("/v1/risk", headers={"X-API-Key": "wrong"}).status_code == 401
    response = client.get("/v1/risk", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    members = response.json()["members"]
    assert members and [m["rank"] for m in members] == list(range(1, len(members) + 1))
    assert all("arm" not in m for m in members)


def test_api_key_must_be_configured(service, monkeypatch):
    client, _ = service
    monkeypatch.delenv("SLACK_ML_API_KEY")
    assert client.get("/v1/risk", headers={"X-API-Key": API_KEY}).status_code == 503


def test_experts_returns_members_with_matching_topic(service):
    client, _ = service
    response = client.get(
        "/v1/experts", params={"q": "pandas で集計", "limit": 2}, headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    assert {m["user_id"] for m in response.json()["members"]} == {"U1", "U2"}


def test_slack_rejects_bad_or_old_signatures(service):
    client, _ = service
    assert slack_request(client, "risk", secret="wrong-secret").status_code == 401
    assert slack_request(client, "risk", timestamp=int(time.time()) - 600).status_code == 401
    assert slack_request(client, "help").status_code == 200


def test_slack_risk_is_only_for_managers_and_is_sent_later(service):
    client, sent = service
    denied = slack_request(client, "risk", user_id="USOMEONE")
    assert "コミュニティマネージャーだけ" in denied.json()["text"]
    assert sent == []

    accepted = slack_request(client, "risk")
    assert accepted.status_code == 200
    assert accepted.json()["response_type"] == "ephemeral"
    [(url, message)] = sent
    assert url == "https://hooks.example/r"
    assert message["response_type"] == "ephemeral"
    assert "今週の候補" in message["text"]


def test_slack_experts_sends_results(service):
    client, sent = service
    slack_request(client, "experts pandas で集計したい")
    [(_, message)] = sent
    assert "詳しそうなメンバー" in message["text"]


def test_verify_signature_function():
    body = b"a=1"
    ts = "1700000000"
    good = "v0=" + hmac.new(b"s", b"v0:1700000000:a=1", hashlib.sha256).hexdigest()
    assert api.verify_slack_signature("s", ts, body, good, now=1700000000)
    assert not api.verify_slack_signature("s", ts, b"a=2", good, now=1700000000)
    assert not api.verify_slack_signature("s", ts, body, good, now=1700000000 + 301)
    assert not api.verify_slack_signature("s", "not-a-number", body, good)
