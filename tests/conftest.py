"""実データの癖を、決まった規則で再現する（乱数は使わないので、毎回同じデータになる）。"""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from slack_ml.config import Config, PathsConfig

JST = timezone(timedelta(hours=9))
LA = ZoneInfo("America/Los_Angeles")
START = datetime(2026, 1, 5, tzinfo=JST)
N_DAYS = 120
STOP_DAY = 60
DUPLICATE_DAY = 2


def _ts(day: int, hour: int, minute: int = 0) -> str:
    return f"{(START + timedelta(days=day, hours=hour, minutes=minute)).timestamp():.6f}"


def build_export(root: Path) -> dict:
    """root の下にエクスポートとメタデータの ZIP を作り、種類ごとの件数を返す。"""
    files: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    expected: Counter = Counter()

    def add(day: int, channel: str, message: dict, kind: str) -> None:
        batch = "batchA/originals" if day < STOP_DAY else "batchB/originals"
        la_date = datetime.fromtimestamp(float(message["ts"]), LA).date().isoformat()
        files[(batch, channel, la_date)].append(message)
        expected[kind] += 1
        expected["bronze_messages"] += 1
        expected["reactions"] += sum(len(r["users"]) for r in message.get("reactions", []))

    for i, user in enumerate(["U1", "U2", "U3", "U4"]):
        join = {"type": "message", "subtype": "channel_join", "user": user, "ts": _ts(0, 8, i)}
        add(0, "general", join, "member_system")
    add(0, "nippo", {"type": "message", "subtype": "channel_join", "user": "UBOT", "ts": _ts(0, 8)}, "bot")
    add(
        30,
        "general",
        {"type": "message", "subtype": "channel_join", "user": "U5", "ts": _ts(30, 8)},
        "member_system",
    )
    add(
        50,
        "general",
        {"type": "message", "subtype": "tombstone", "user": "USLACKBOT", "ts": _ts(50, 5)},
        "bot",
    )

    for d in range(N_DAYS):
        report = {"type": "message", "user": "UBOT", "bot_id": "B1", "ts": _ts(d, 9), "text": "日報"}
        add(d, "nippo", report, "bot")
        add(
            d,
            "nippo",
            {
                "type": "message",
                "user": "UBOT",
                "bot_id": "B1",
                "ts": _ts(d, 9, 1),
                "thread_ts": report["ts"],
            },
            "bot",
        )
        if d % 7 == 0:
            comment = {
                "type": "message",
                "user": "U2",
                "ts": _ts(d, 9, 30),
                "thread_ts": report["ts"],
                "text": "見ました",
            }
            add(d, "nippo", comment, "member_activity")

        post = {"type": "message", "user": "U1", "ts": _ts(d, 10), "text": f"今日の作業 {d}"}
        if d % 2 == 0:
            post["reactions"] = [{"name": "eyes", "users": ["U3"], "count": 1}]
        add(d, "general", post, "member_activity")
        if d + 1 < N_DAYS:
            reply = {
                "type": "message",
                "user": "U2",
                "ts": _ts(d + 1, 10, 30),
                "thread_ts": post["ts"],
                "text": "いいですね",
            }
            add(d + 1, "general", reply, "member_activity")
        if d % 4 == 0:
            note = {
                "type": "message",
                "user": "U1",
                "ts": _ts(d, 11),
                "thread_ts": post["ts"],
                "text": "追記",
            }
            add(d, "general", note, "member_activity")

        if d < STOP_DAY and d % 3 == 0:
            u4 = {
                "type": "message",
                "user": "U4",
                "ts": _ts(d, 20),
                "text": "質問です",
                "reactions": [{"name": "+1", "users": ["U1"], "count": 1}],
            }
            add(d, "general", u4, "member_activity")
        if d >= 30 and (d - 30) % 7 == 0:
            add(
                d,
                "general",
                {"type": "message", "user": "U5", "ts": _ts(d, 21), "text": "週報"},
                "member_activity",
            )
        if d % 10 == 0:
            app = {"type": "message", "user": "U2", "bot_id": "BAPP", "ts": _ts(d, 7), "text": "自動投稿"}
            add(d, "general", app, "member_app")
        if d % 15 == 0:
            add(
                d,
                "nippo",
                {"type": "message", "subtype": "bot_message", "bot_id": "B1", "ts": _ts(d, 6)},
                "no_user",
            )

    duplicated = next(
        msg
        for msgs in files.values()
        for msg in msgs
        if msg.get("user") == "U1" and msg["ts"] == _ts(DUPLICATE_DAY, 10)
    )
    la_date = datetime.fromtimestamp(float(duplicated["ts"]), LA).date().isoformat()
    files[("batchB/originals", "general", la_date)].append(duplicated)
    expected["bronze_messages"] += 1

    for (batch, channel, la_date), messages in files.items():
        folder = root / "export" / batch / channel
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{la_date}.json").write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")

    users = [
        {"id": f"U{i}", "name": f"user{i}", "real_name": f"User {i}", "profile": {}} for i in range(1, 7)
    ]
    users += [
        {"id": "UBOT", "name": "bot", "is_bot": True, "profile": {}},
        {"id": "USLACKBOT", "name": "slackbot", "is_bot": False, "profile": {}},
    ]
    channels = [{"id": "C1", "name": "general", "is_general": True}, {"id": "C2", "name": "nippo"}]
    with zipfile.ZipFile(root / "meta.zip", "w") as z:
        z.writestr("ws/users.json", json.dumps(users))
        z.writestr("ws/channels.json", json.dumps(channels))
    return dict(expected)


@pytest.fixture
def export(tmp_path: Path) -> tuple[Config, dict]:
    """テスト用のエクスポートを作り、それを読む設定と、期待する件数を返す。"""
    expected = build_export(tmp_path)
    paths = PathsConfig(
        export_dir=tmp_path / "export",
        metadata_export=tmp_path / "meta.zip",
        warehouse=tmp_path / "warehouse.duckdb",
    )
    return Config(paths=paths), expected
