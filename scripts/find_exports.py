"""展開したフォルダの中から「エクスポートの場所」を探し、件数と、場所どうしの重なりを比べる。

使い方:
    python scripts/find_exports.py data/real/extracted
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DAY_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def scan(root: Path) -> dict[Path, dict]:
    exports: dict[Path, dict] = defaultdict(
        lambda: {"channels": set(), "files": 0, "dates": [], "keys": set()}
    )
    for f in root.rglob("*.json"):
        if not DAY_FILE.match(f.name):
            continue
        e = exports[f.parent.parent]
        e["channels"].add(f.parent.name)
        e["files"] += 1
        e["dates"].append(f.stem)
        for m in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(m, dict) and "ts" in m:
                e["keys"].add((f.parent.name, str(m["ts"])))
    return exports


def main(root: Path) -> None:
    exports = scan(root)
    names = sorted(exports)
    print("== エクスポートの場所ごとの件数 ==")
    for d in names:
        e = exports[d]
        print(
            f"{d.relative_to(root)}: チャンネル {len(e['channels'])} / 日別ファイル {e['files']}"
            f" / メッセージ {len(e['keys'])} / 期間 {min(e['dates'])} 〜 {max(e['dates'])}"
        )

    print("\n== 場所どうしの重なり（メッセージ数） ==")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka, kb = exports[a]["keys"], exports[b]["keys"]
            print(
                f"{a.relative_to(root)} と {b.relative_to(root)}: 共通 {len(ka & kb)}"
                f" / 前者だけ {len(ka - kb)} / 後者だけ {len(kb - ka)}"
            )

    all_keys = set().union(*(exports[d]["keys"] for d in names))
    print(f"\n重複を除いたメッセージ数（全体）: {len(all_keys)}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/real/extracted"))
