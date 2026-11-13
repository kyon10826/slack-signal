"""Slack エクスポート（チャンネルごとのフォルダに日別の JSON があるもの）を読み、表に変換する。"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pandas as pd

log = logging.getLogger(__name__)

DAY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
KNOWN_MESSAGE_KEYS = {
    "type",
    "subtype",
    "user",
    "text",
    "ts",
    "thread_ts",
    "parent_user_id",
    "reply_count",
    "reply_users",
    "reply_users_count",
    "latest_reply",
    "reactions",
    "bot_id",
    "bot_profile",
    "app_id",
    "username",
    "edited",
    "files",
    "attachments",
    "blocks",
    "client_msg_id",
    "team",
    "user_team",
    "source_team",
    "user_profile",
    "is_locked",
    "subscribed",
    "last_read",
    "upload",
    "display_as_bot",
}
USER_COLUMNS = ["user_id", "name", "real_name", "display_name", "is_bot", "deleted"]
CHANNEL_COLUMNS = ["channel_id", "channel_name", "is_general"]
MESSAGE_COLUMNS = [
    "channel_id",
    "channel_name",
    "ts",
    "ts_epoch",
    "user_id",
    "subtype",
    "bot_id",
    "text",
    "thread_ts",
    "parent_user_id",
    "source_file",
]
REACTION_COLUMNS = ["channel_id", "message_ts", "reaction", "user_id"]
QUARANTINE_COLUMNS = ["source_file", "reason", "payload"]


@dataclass
class ExportTables:
    """エクスポートを表にした結果。"""

    users: pd.DataFrame
    channels: pd.DataFrame
    messages: pd.DataFrame
    reactions: pd.DataFrame
    quarantine: pd.DataFrame
    unknown_keys: Counter = field(default_factory=Counter)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "users": len(self.users),
            "channels": len(self.channels),
            "messages": len(self.messages),
            "reactions": len(self.reactions),
            "quarantined": len(self.quarantine),
            "unknown_keys": sum(self.unknown_keys.values()),
        }


def find_channel_dirs(export_dir: Path) -> dict[str, list[Path]]:
    """export_dir の下を深さに関係なく探し、日別ファイルを含むフォルダをチャンネルとみなす。"""
    found: dict[str, list[Path]] = defaultdict(list)
    for d in sorted({f.parent for f in export_dir.rglob("*.json") if DAY_FILE.match(f.name)}):
        found[d.name].append(d)
    return dict(sorted(found.items()))


@dataclass
class _UserEvidence:
    """メッセージを読みながら、メンバー一覧を作るための材料を集める。"""

    profiles: dict[str, tuple[float, dict]] = field(default_factory=dict)
    human_posts: Counter = field(default_factory=Counter)
    bot_posts: Counter = field(default_factory=Counter)
    seen: set[str] = field(default_factory=set)

    def add_message(self, m: dict, ts_epoch: float) -> None:
        uid = m.get("user")
        if not uid:
            return
        self.seen.add(uid)
        if m.get("bot_id") or m.get("subtype") == "bot_message":
            self.bot_posts[uid] += 1
        else:
            self.human_posts[uid] += 1
        profile = m.get("user_profile")
        if profile and (uid not in self.profiles or ts_epoch > self.profiles[uid][0]):
            self.profiles[uid] = (ts_epoch, profile)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for uid in sorted(self.seen):
            profile = self.profiles.get(uid, (0.0, {}))[1]
            rows.append(
                {
                    "user_id": uid,
                    "name": profile.get("name"),
                    "real_name": profile.get("real_name") or profile.get("display_name") or uid,
                    "display_name": profile.get("display_name"),
                    "is_bot": uid == "USLACKBOT" or (self.bot_posts[uid] > 0 and self.human_posts[uid] == 0),
                    "deleted": False,
                }
            )
        return pd.DataFrame(rows, columns=USER_COLUMNS)


def _load_day_file(path: Path, rel: str, quarantine: list[dict]) -> list[dict] | None:
    """日別ファイルを読む。壊れていたら隔離して None を返す。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        quarantine.append({"source_file": rel, "reason": f"invalid_json: {e}", "payload": None})
        return None
    if not isinstance(data, list):
        quarantine.append({"source_file": rel, "reason": "not_a_list", "payload": None})
        return None
    return data


def read_metadata(zip_path: Path) -> tuple[list[dict] | None, list[dict] | None]:
    """公式エクスポートの ZIP から users.json と channels.json を読む。無いものは None を返す。"""
    with zipfile.ZipFile(zip_path) as z:

        def load(filename: str) -> list[dict] | None:
            hits = [n for n in z.namelist() if PurePosixPath(n).name == filename]
            return json.loads(z.read(min(hits, key=lambda n: n.count("/")))) if hits else None

        return load("users.json"), load("channels.json")


def _users_from_json(raw: list[dict]) -> pd.DataFrame:
    rows = []
    for u in raw:
        profile = u.get("profile") or {}
        rows.append(
            {
                "user_id": u.get("id"),
                "name": u.get("name"),
                "real_name": u.get("real_name") or profile.get("real_name") or u.get("name"),
                "display_name": profile.get("display_name"),
                "is_bot": bool(u.get("is_bot")) or u.get("id") == "USLACKBOT",
                "deleted": bool(u.get("deleted")),
            }
        )
    return pd.DataFrame(rows, columns=USER_COLUMNS)


def read_export(
    export_dir: Path, metadata: Path | None = None, exclude_users: tuple[str, ...] = ()
) -> ExportTables:
    """エクスポートのフォルダを読み、5つの表にして返す。"""
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError(
            f"エクスポートのフォルダが見つかりません: {export_dir}（tar.gz や zip は展開してから指定してください）"
        )
    channel_dirs = find_channel_dirs(export_dir)
    channel_names = list(channel_dirs)
    if not channel_names:
        raise ValueError(f"日別ファイル（YYYY-MM-DD.json）を含むフォルダがありません: {export_dir}")

    users_json, channels_json = read_metadata(metadata) if metadata is not None else (None, None)
    channel_meta = {c.get("name"): c for c in channels_json or []}
    channel_ids = {name: channel_meta.get(name, {}).get("id") or name for name in channel_names}

    excluded = set(exclude_users)
    messages: list[dict] = []
    reactions: list[dict] = []
    quarantine: list[dict] = []
    unknown: Counter = Counter()
    evidence = _UserEvidence()
    joiners: dict[str, set[str]] = defaultdict(set)

    day_files = [
        (channel, path)
        for channel, dirs in channel_dirs.items()
        for d in dirs
        for path in sorted(d.iterdir())
        if DAY_FILE.match(path.name)
    ]
    for channel, path in day_files:
        rel = path.relative_to(export_dir).as_posix()
        day_msgs = _load_day_file(path, rel, quarantine)
        if day_msgs is None:
            continue
        for m in day_msgs:
            if "ts" not in m:
                payload = json.dumps(m, ensure_ascii=False)[:2000]
                quarantine.append({"source_file": rel, "reason": "missing_ts", "payload": payload})
                continue
            unknown.update(k for k in m if k not in KNOWN_MESSAGE_KEYS)
            user_id = m.get("user")
            if user_id in excluded:
                continue
            try:
                ts_epoch = float(m["ts"])
            except (TypeError, ValueError):
                quarantine.append({"source_file": rel, "reason": "invalid_ts", "payload": str(m["ts"])})
                continue

            evidence.add_message(m, ts_epoch)
            if m.get("subtype") == "channel_join" and user_id:
                joiners[channel].add(user_id)
            messages.append(
                {
                    "channel_id": channel_ids[channel],
                    "channel_name": channel,
                    "ts": str(m["ts"]),
                    "ts_epoch": ts_epoch,
                    "user_id": user_id,
                    "subtype": m.get("subtype"),
                    "bot_id": m.get("bot_id"),
                    "text": m.get("text") or "",
                    "thread_ts": m.get("thread_ts"),
                    "parent_user_id": m.get("parent_user_id"),
                    "source_file": rel,
                }
            )
            for r in m.get("reactions") or []:
                for reactor in r.get("users") or []:
                    if reactor in excluded:
                        continue
                    evidence.seen.add(reactor)
                    reactions.append(
                        {
                            "channel_id": channel_ids[channel],
                            "message_ts": str(m["ts"]),
                            "reaction": r.get("name"),
                            "user_id": reactor,
                        }
                    )

    locations = {d.parent for dirs in channel_dirs.values() for d in dirs}
    notes: list[str] = [f"{len(locations)} か所のフォルダから {len(day_files)} 個の日別ファイルを読んだ"]
    split = sum(len(dirs) > 1 for dirs in channel_dirs.values())
    if split:
        notes.append(f"{split} チャンネルは複数のフォルダに分かれていたので、同じチャンネルとしてまとめた")
    inferred = evidence.to_frame()
    if users_json is not None:
        listed = _users_from_json(users_json)
        missing = inferred[~inferred["user_id"].isin(listed["user_id"])]
        users = pd.concat([listed, missing], ignore_index=True)
        notes.append(
            f"users.json から {len(listed)} 人を読み、載っていない {len(missing)} 人をメッセージから補った"
        )
    else:
        users = inferred
        notes.append(f"メンバー一覧が無いため、メッセージから {len(users)} 人を推定した")
    users = users[~users["user_id"].isin(excluded)].reset_index(drop=True)

    channels = pd.DataFrame(
        {
            "channel_id": [channel_ids[c] for c in channel_names],
            "channel_name": channel_names,
            "is_general": [bool(channel_meta.get(c, {}).get("is_general")) for c in channel_names],
        },
        columns=CHANNEL_COLUMNS,
    )
    matched = sum(c in channel_meta for c in channel_names)
    if channels_json is not None:
        notes.append(f"{len(channel_names)} チャンネルのうち {matched} を channels.json と名前で対応づけた")
    else:
        notes.append(f"チャンネル一覧が無いため、フォルダ名から {len(channel_names)} チャンネルとみなした")
    if not channels["is_general"].any() and joiners:
        general = max(joiners, key=lambda c: (len(joiners[c]), c == "general"))
        channels.loc[channels["channel_name"] == general, "is_general"] = True
        notes.append(f"参加ログが最も多い #{general} を #general とみなした")

    for note in notes:
        log.info(note)
    if quarantine:
        log.warning("隔離したレコード: %d 件", len(quarantine))

    return ExportTables(
        users=users,
        channels=channels,
        messages=pd.DataFrame(messages, columns=MESSAGE_COLUMNS),
        reactions=pd.DataFrame(reactions, columns=REACTION_COLUMNS),
        quarantine=pd.DataFrame(quarantine, columns=QUARANTINE_COLUMNS),
        unknown_keys=unknown,
        notes=notes,
    )
