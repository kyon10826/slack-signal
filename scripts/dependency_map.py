"""属人化マップと副担当の候補を表示する（gold.person_dependency、channel_dependency、backup_candidates）。

使い方:
    python scripts/dependency_map.py
    python scripts/dependency_map.py --names
    python scripts/dependency_map.py --names configs/default.yaml
"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

TOP_PEOPLE = 10

show_names = "--names" in sys.argv
args = [a for a in sys.argv[1:] if a != "--names"]
cfg = load_config(args[0] if args else "configs/local.yaml")
managers = set(cfg.slack.manager_user_ids)

con = wh.connect(cfg.paths.warehouse, read_only=True)
people_rows = con.execute(
    """
    SELECT user_id, solo, solo_share, channels, dependent_channels, heavy_channels
    FROM gold.person_dependency
    WHERE solo > 0
    ORDER BY solo_rank, user_id
    LIMIT ?
    """,
    [TOP_PEOPLE],
).fetchall()
[(total_solo, solo_people)] = con.execute(
    "SELECT COALESCE(SUM(solo), 0), COUNT(*) FILTER (WHERE solo > 0) FROM gold.person_dependency"
).fetchall()
channels = con.execute(
    """
    SELECT channel_id, channel_name, threads, responders, key_person_id, key_solo, key_share, solo_share,
           is_dependent, is_heavy
    FROM gold.channel_dependency
    WHERE threads >= 5
    ORDER BY is_dependent DESC, key_solo DESC, solo_share DESC
    """
).fetchall()
[(skipped,)] = con.execute("SELECT COUNT(*) FROM gold.channel_dependency WHERE threads < 5").fetchall()
candidates = con.execute(
    """
    SELECT channel_id, candidate_id, answered, shared_threads, messages, is_loaded, is_taken
    FROM gold.backup_candidates
    ORDER BY channel_id, candidate_rank
    """
).fetchall()
names = dict(
    con.execute(
        "SELECT user_id, COALESCE(NULLIF(display_name, ''), NULLIF(real_name, ''), name, user_id) FROM silver.users"
    ).fetchall()
)
con.close()

people: dict[str, str] = {}
rooms: dict[str, str] = {}


def who(user_id: str) -> str:
    """人の表示名。--names が無いときは、出てきた順に「人物1」「人物2」と伏せる。"""
    if show_names:
        label = names.get(user_id, user_id)
    else:
        label = people.setdefault(user_id, f"人物{len(people) + 1}")
    return label + ("（部署長）" if user_id in managers else "")


def where(channel_id: str, channel_name: str | None) -> str:
    """チャンネルの表示名。--names が無いときは「チャンネル1」のように伏せる。"""
    if show_names:
        return f"#{channel_name or channel_id}"
    return rooms.setdefault(channel_id, f"チャンネル{len(rooms) + 1}")


print("== 人ごとの負担（その人が休むと答えが付かなくなるスレッドの合計。直近90日） ==")
print(f"  答えた人が1人だけのスレッド {total_solo}件を、{solo_people}人で分けている\n")
cumulative = 0.0
for rank, (user_id, solo, share, n_channels, n_dependent, n_heavy) in enumerate(people_rows, start=1):
    cumulative += share
    roles = []
    if n_dependent:
        roles.append(f"属人化 {n_dependent}")
    if n_heavy:
        roles.append(f"件数が多い {n_heavy}")
    role = f" · 担当: {'、'.join(roles)}" if roles else ""
    print(
        f"  {rank:>2}. {who(user_id):<10} ひとりで答えた {solo:>3}件（全体の {share:>4.0%}、上位からの累計 "
        f"{cumulative:>4.0%}） · 答えたチャンネル {n_channels}{role}"
    )

dependent = [c for c in channels if c[8]]
many = [c for c in channels if c[9]]
print("\n== 属人化マップ（チャンネルごと。答えが付いたスレッドが5件以上） ==")
print(
    f"  判断したチャンネル {len(channels)} / 属人化している {len(dependent)} / "
    f"割合は半分以下だが、ひとりで答えた件数が多い {len(many)}（5件未満で判断しなかった {skipped}）\n"
)
for channel_id, name, threads, responders, key, key_solo, key_share, solo_share, is_dep, is_heavy in channels:
    mark = "  ← 属人化" if is_dep else ("  ← 件数が多い" if is_heavy else "")
    print(
        f"  {where(channel_id, name):<14} スレッド {threads:>4}件  答えた人 {responders:>2}人  "
        f"{who(key)} がひとりで答えた {key_solo:>3}件（{solo_share:>4.0%}、関わった {key_share:.0%}）{mark}"
    )

print("\n== 副担当の候補（属人化しているチャンネル。直近28日に行動した人から3人まで） ==")
if not dependent:
    print("  属人化しているチャンネルはありません")
by_channel: dict[str, list[tuple]] = {}
for channel_id, *rest in candidates:
    by_channel.setdefault(channel_id, []).append(tuple(rest))
for channel_id, name, _threads, _responders, key, *_ in dependent:
    print(f"\n  {where(channel_id, name)}（担当: {who(key)}）")
    rows = by_channel.get(channel_id, [])
    if not rows:
        print("    候補がいません（このチャンネルで最近発言している人がいない）")
    for rank, (candidate, answered, shared, messages, is_loaded, is_taken) in enumerate(rows, start=1):
        notes = []
        if is_loaded:
            notes.append("すでに担当者として負担あり")
        if is_taken:
            notes.append("件数がもっと多いチャンネルでも候補")
        note = f" · {'、'.join(notes)}" if notes else ""
        print(
            f"    {rank}. {who(candidate)}  このチャンネルで答えた {answered}件 · "
            f"担当者と同じスレッド {shared}件 · 発言 {messages}件{note}"
        )

if many:
    print("\n== 割合は半分以下だが、ひとりで答えた件数が多いチャンネル ==")
    print("  属人化の定義（半分より多い）には入らないので、副担当の候補は出していない。人が見て判断する")
    for channel_id, name, _threads, _responders, key, key_solo, *_ in many:
        print(f"  {where(channel_id, name)}（{who(key)} がひとりで {key_solo}件）")
