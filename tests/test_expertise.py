"""詳しい人探し（expertise）のテスト。話題のはっきりした小さな発言データで確かめる。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from slack_ml import churn, expertise, pipeline, scoring
from slack_ml import warehouse as wh

TOPICS = {
    "UPY1": "pandas と numpy で集計を速くする方法。pytest でテストも書いた",
    "UPY2": "numpy の配列計算と pandas の groupby の使い方。型ヒントも付けた",
    "UDS1": "Figma で UI デザインの配色を決めた。ワイヤーフレームとプロトタイプも作った",
    "UCR1": "職務経歴書の書き方と面接の準備。転職の相談に乗ってほしい",
}


def messages_frame(start: date = date(2026, 1, 5), days: int = 20) -> pd.DataFrame:
    """メンバーごとに同じ話題の投稿を毎日1件ずつ作る。UPY1 の投稿には UPY2 が返信する。"""
    rows = []
    for d in range(days):
        day = start + timedelta(days=d)
        for user, text in TOPICS.items():
            ts = f"{d}.{user}"
            rows.append(["C1", ts, user, day, False, None, f"{text} {d}日目"])
            if user == "UPY1":
                rows.append(["C1", f"{ts}.r", "UPY2", day, True, ts, "numpy ならこう書けます"])
    df = pd.DataFrame(
        rows, columns=["channel_id", "ts", "user_id", "posted_date", "is_reply", "thread_ts", "text"]
    )
    df["clean"] = df["text"].map(expertise.clean_text)
    return df


def test_clean_text_removes_slack_markup_but_keeps_link_labels():
    text = "<@U123> 見て <https://example.com|良い記事> :smile: <https://x.com> <#C1|general> <!here>"
    assert expertise.clean_text(text) == "見て 良い記事"
    assert expertise.clean_text(None) == ""


def test_search_returns_members_with_matching_topic():
    messages = messages_frame()
    index = expertise.build_index(messages, until=messages["posted_date"].max() + timedelta(days=1))
    top = [user for user, _ in index.search("pandas で groupby の集計をしたい", top_n=2)]
    assert set(top) == {"UPY1", "UPY2"}
    assert index.search("Figma の配色", top_n=1)[0][0] == "UDS1"
    assert "UPY1" not in [user for user, _ in index.search("pandas", top_n=3, exclude={"UPY1"})]


def test_similar_members_and_candidates():
    messages = messages_frame()
    index = expertise.build_index(messages, until=messages["posted_date"].max() + timedelta(days=1))
    assert index.similar("UPY1", top_n=1)[0][0] == "UPY2"
    assert index.similar("UPY1", top_n=3, candidates={"UDS1"})[0][0] == "UDS1"
    assert index.similar("UNKNOWN") == []


def test_index_is_none_when_members_share_no_words():
    """共通する語がほとんど無いと、話題の近さを測れないので None を返す（落ちずに、つなぐ相手が空になる）。"""
    messages = pd.DataFrame(
        {
            "user_id": ["UA", "UB"],
            "posted_date": [date(2026, 1, 5)] * 2,
            "clean": ["あいう", "かきく"],
        }
    )
    assert expertise.build_index(messages, until=date(2026, 1, 6), min_messages=1) is None
    assert expertise.suggest_partners(None, pd.DataFrame(), k=5).empty


def test_index_uses_only_messages_before_until():
    """索引は until より前の発言だけで作る（評価のときに、未来の発言を使わないため）。"""
    messages = messages_frame()
    until = messages["posted_date"].min() + timedelta(days=5)
    index = expertise.build_index(messages, until=until, min_messages=1)
    assert index.built_until == until
    assert all(count == 5 for user, count in index.message_counts.items() if user != "UPY2")


def test_evaluate_finds_repliers_better_than_random_on_topic_data():
    result = expertise.evaluate(messages_frame(days=40), eval_days=10, top_n=1, min_messages=3)
    assert result["queries"] > 0
    assert result["hit@1_model"] == 1.0
    assert result["hit@1_model"] > result["hit@1_random"]


def test_partners_are_active_and_not_in_the_top_list(export):
    """つなぐ相手は、直近7日に行動していて、今週声をかけたいメンバーに入っていない人から選ぶ。"""
    cfg, _ = export
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse)
    labeled = churn.load_labeled(con)
    scored = scoring.score_latest(labeled, scoring.load_latest(con), "logistic")
    messages = expertise.load_messages(con)
    topics = list(TOPICS.values())
    topic_of = {user: topics[i % len(topics)] for i, user in enumerate(sorted(messages["user_id"].unique()))}
    messages["clean"] = messages["user_id"].map(topic_of) + " " + messages["clean"]
    index = expertise.build_index(messages, messages["posted_date"].max() + timedelta(days=1), min_messages=1)
    k = 1
    partners = expertise.suggest_partners(index, scored, k, top_n=2)
    expertise.save_partners(con, partners, scored["snapshot_date"].iloc[0])
    expertise.save_partners(con, partners, scored["snapshot_date"].iloc[0])
    [(saved,)] = con.execute("SELECT COUNT(*) FROM ops.connect_suggestions").fetchall()
    con.close()

    top = set(scored[scored["early"]].sort_values("rank_early").head(k)["user_id"])
    active = set(scored.loc[scored["early"], "user_id"])
    assert len(partners) > 0
    assert saved == len(partners)
    assert set(partners["user_id"]) <= top
    assert set(partners["partner_id"]) <= active - top
