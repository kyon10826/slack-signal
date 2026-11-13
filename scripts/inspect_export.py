"""Slack エクスポート（tar.gz / zip / フォルダ）の構造を確かめる"""

from __future__ import annotations

import json
import re
import sys
import tarfile
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path, PurePosixPath

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SPECIAL_FILES = {
    "users.json",
    "channels.json",
    "groups.json",
    "dms.json",
    "mpims.json",
    "integration_logs.json",
}


def iter_files(path: Path) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """(アーカイブ内のパス, 中身を読む関数) を1件ずつ返す。ディスクには展開しない。"""
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                yield p.relative_to(path).as_posix(), p.read_bytes
    elif zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for name in z.namelist():
            if not name.endswith("/"):
                yield name, (lambda n=name: z.read(n))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if m.isfile():
                    yield m.name, (lambda m=m: t.extractfile(m).read())
    else:
        raise SystemExit(f"フォルダ、zip、tar のどれでもありません: {path}")


def main(path_str: str) -> None:
    found: Counter = Counter()
    other: Counter = Counter()
    day_files: Counter = Counter()
    by_month: Counter = Counter()
    subtypes: Counter = Counter()
    keys: Counter = Counter()
    dates: list[str] = []
    n_msgs = n_replies = n_with_reactions = 0
    users_info = channels_info = ""

    for name, read in iter_files(Path(path_str)):
        p = PurePosixPath(name)
        if p.name in SPECIAL_FILES:
            found[p.name] += 1
            data = json.loads(read())
            if p.name == "users.json":
                bots = sum(1 for u in data if u.get("is_bot"))
                deleted = sum(1 for u in data if u.get("deleted"))
                users_info = f"{len(data)}人（うちボット {bots}、退会済み {deleted}）"
            elif p.name == "channels.json":
                channels_info = f"{len(data)}チャンネル"
        elif p.suffix == ".json" and DATE.match(p.stem):
            msgs = json.loads(read())
            day_files[p.parent.name] += 1
            dates.append(p.stem)
            by_month[p.stem[:7]] += len(msgs)
            for m in msgs:
                n_msgs += 1
                subtypes[m.get("subtype") or "(通常の発言)"] += 1
                keys.update(m.keys())
                if m.get("thread_ts") and m.get("thread_ts") != m.get("ts"):
                    n_replies += 1
                if m.get("reactions"):
                    n_with_reactions += 1
        else:
            other[p.suffix or p.name] += 1

    print("== ファイル構成 ==")
    for f in sorted(SPECIAL_FILES):
        print(f"  {f:22s} {'あり' if found[f] else 'なし'}")
    print(f"  ユーザー: {users_info or '不明'} / チャンネル一覧: {channels_info or '不明'}")
    print(f"  日別ファイルのあるチャンネル: {len(day_files)} / 日別ファイル: {sum(day_files.values())}")
    print(f"  その他のファイル: {dict(other.most_common(8))}")

    if not dates:
        print("日付ごとの JSON が見つかりません。Slack 公式エクスポートとは別の形式の可能性があります")
        return
    first, last = date.fromisoformat(min(dates)), date.fromisoformat(max(dates))
    span = (last - first).days + 1
    print("\n== 期間とメッセージ ==")
    print(f"  期間: {first} 〜 {last}（{span}日）")
    print(f"  メッセージ: {n_msgs}件（スレッド返信 {n_replies}件、リアクション付き {n_with_reactions}件）")
    print("  月別のメッセージ数:")
    for month in sorted(by_month):
        print(f"    {month}: {by_month[month]}")
    print(f"  subtype 上位: {dict(subtypes.most_common(8))}")
    print(f"  メッセージの項目 上位: {[k for k, _ in keys.most_common(15)]}")

    print("\n== 離脱予測モデルに足りるか ==")
    print(f"  デフォルト設定（約266日必要）: {'足りる' if span >= 266 else '足りない'}")
    print(f"  小さめの設定（約182日必要）  : {'足りる' if span >= 182 else '足りない'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("使い方: python scripts/inspect_export.py <エクスポートのパス>")
    main(sys.argv[1])
