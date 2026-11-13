"""詳しい人探し: メンバーごとに発言の話題をベクトルにし、質問文に近いメンバーを返す。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from math import comb
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MENTION = re.compile(r"<[@!#][^>]*>")
LINK_WITH_LABEL = re.compile(r"<https?://[^|>]+\|([^>]+)>")
LINK = re.compile(r"<https?://[^>]+>|https?://\S+")
EMOJI = re.compile(r":[a-z0-9_+\-]+:")
SPACES = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Slack の記法（メンション、リンク、絵文字）を取り除き、空白をまとめる。リンクの表示名は残す。"""
    if not text:
        return ""
    text = LINK_WITH_LABEL.sub(r"\1", text)
    text = LINK.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = EMOJI.sub(" ", text)
    return SPACES.sub(" ", text).strip()


def load_messages(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """活動に数える発言（メンバーの投稿と返信）を、本文の前処理をして読む。"""
    df = con.execute(
        """
        SELECT channel_id, ts, user_id, posted_date, is_reply, thread_ts, text
        FROM silver.messages
        WHERE is_activity AND text IS NOT NULL
        ORDER BY posted_at
        """
    ).df()
    df["posted_date"] = pd.to_datetime(df["posted_date"]).dt.date
    df["clean"] = df["text"].map(clean_text)
    return df[df["clean"] != ""].reset_index(drop=True)


@dataclass
class ExpertIndex:
    """メンバーごとの話題のベクトル。search で質問文に近いメンバーを、similar で話題の近いメンバーを返す。"""

    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    user_ids: list[str]
    vectors: np.ndarray
    message_counts: dict[str, int]
    built_until: date

    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(self.svd.transform(self.vectorizer.transform([clean_text(t) for t in texts])))

    def _rank(self, scores: np.ndarray, top_n: int, exclude, candidates) -> list[tuple[str, float]]:
        allowed = set(candidates) if candidates is not None else None
        order = np.argsort(-scores, kind="stable")
        result = []
        for i in order:
            user = self.user_ids[i]
            if user in exclude or (allowed is not None and user not in allowed):
                continue
            result.append((user, float(scores[i])))
            if len(result) == top_n:
                break
        return result

    def search(self, query: str, top_n: int = 5, exclude=(), candidates=None) -> list[tuple[str, float]]:
        """質問文に話題の近いメンバーを、近い順に (user_id, 類似度) で返す。"""
        scores = self.vectors @ self.encode([query])[0]
        return self._rank(scores, top_n, set(exclude), candidates)

    def similar(self, user_id: str, top_n: int = 3, exclude=(), candidates=None) -> list[tuple[str, float]]:
        """あるメンバーと話題の近いメンバーを返す（本人は除く）。索引に無いメンバーなら空のリスト。"""
        if user_id not in self.user_ids:
            return []
        scores = self.vectors @ self.vectors[self.user_ids.index(user_id)]
        return self._rank(scores, top_n, set(exclude) | {user_id}, candidates)


def build_index(
    messages: pd.DataFrame,
    until: date,
    lookback_days: int = 120,
    min_messages: int = 3,
    n_components: int = 64,
    random_state: int = 42,
) -> ExpertIndex | None:
    """until の前日までの lookback_days 日の発言で索引を作る。"""
    window = messages[
        (messages["posted_date"] < until) & (messages["posted_date"] >= until - timedelta(days=lookback_days))
    ]
    counts = window.groupby("user_id").size()
    users = sorted(counts[counts >= min_messages].index)
    if len(users) < 2:
        return None
    docs = window[window["user_id"].isin(users)].groupby("user_id")["clean"].apply(" ".join).loc[users]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2, sublinear_tf=True)
    try:
        tfidf = vectorizer.fit_transform(docs.tolist())
    except ValueError:
        return None
    if tfidf.shape[1] < 2:
        return None
    k = max(1, min(n_components, len(users) - 1, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=random_state)
    vectors = normalize(svd.fit_transform(tfidf))
    return ExpertIndex(
        vectorizer=vectorizer,
        svd=svd,
        user_ids=list(users),
        vectors=vectors,
        message_counts={u: int(counts[u]) for u in users},
        built_until=until,
    )


def evaluation_queries(messages: pd.DataFrame, start: date, min_length: int = 10) -> pd.DataFrame:
    """評価に使う質問文: start 以降のトップレベルの投稿で、他のメンバーから返信が付いたもの。"""
    posts = messages[(~messages["is_reply"]) & (messages["posted_date"] >= start)]
    posts = posts[posts["clean"].str.len() >= min_length]
    replies = messages[messages["is_reply"]][["channel_id", "thread_ts", "user_id"]]
    joined = posts.merge(
        replies.rename(columns={"thread_ts": "ts", "user_id": "replier"}), on=["channel_id", "ts"]
    )
    joined = joined[joined["replier"] != joined["user_id"]]
    return (
        joined.groupby(["channel_id", "ts", "user_id", "clean"])["replier"]
        .apply(lambda s: sorted(set(s)))
        .reset_index()
        .rename(columns={"replier": "repliers"})
    )


def evaluate(
    messages: pd.DataFrame,
    eval_days: int = 56,
    top_n: int = 5,
    lookback_days: int = 120,
    min_messages: int = 3,
    n_components: int = 64,
    random_state: int = 42,
) -> dict:
    """最後の eval_days 日を評価期間にして、hit@N をモデル・ランダム・発言の多い順で比べる。"""
    last = messages["posted_date"].max()
    start = last - timedelta(days=eval_days - 1)
    index = build_index(messages, start, lookback_days, min_messages, n_components, random_state)
    if index is None:
        return {"queries": 0}
    by_activity = sorted(index.user_ids, key=lambda u: -index.message_counts[u])
    hits = {"model": 0, "activity": 0}
    random_hit = 0.0
    used = 0
    for row in evaluation_queries(messages, start).itertuples(index=False):
        candidates = [u for u in index.user_ids if u != row.user_id]
        repliers = set(row.repliers) & set(candidates)
        if not repliers:
            continue
        used += 1
        found = {u for u, _ in index.search(row.clean, top_n, exclude={row.user_id})}
        hits["model"] += bool(found & repliers)
        hits["activity"] += bool(set([u for u in by_activity if u != row.user_id][:top_n]) & repliers)
        n, r = len(candidates), len(repliers)
        random_hit += 1 - comb(n - r, min(top_n, n)) / comb(n, min(top_n, n))
    if used == 0:
        return {"queries": 0}
    return {
        "queries": used,
        "members": len(index.user_ids),
        "top_n": top_n,
        "eval_start": start,
        f"hit@{top_n}_model": hits["model"] / used,
        f"hit@{top_n}_activity": hits["activity"] / used,
        f"hit@{top_n}_random": random_hit / used,
    }


def suggest_partners(index: ExpertIndex | None, scored: pd.DataFrame, k: int, top_n: int = 3) -> pd.DataFrame:
    """今週声をかけたいメンバー（早期発見の上位 k 人）それぞれに、話題の近い「つなぐ相手」を選ぶ。"""
    columns = ["snapshot_date", "user_id", "partner_id", "similarity", "partner_rank"]
    if index is None or scored.empty:
        return pd.DataFrame(columns=columns)
    top = scored[scored["early"]].sort_values("rank_early").head(k)
    candidates = set(scored.loc[scored["early"], "user_id"]) - set(top["user_id"])
    rows = []
    for member in top.itertuples(index=False):
        for rank, (partner, similarity) in enumerate(
            index.similar(member.user_id, top_n, candidates=candidates), start=1
        ):
            rows.append([member.snapshot_date, member.user_id, partner, similarity, rank])
    return pd.DataFrame(rows, columns=columns)


def save_partners(con: duckdb.DuckDBPyConnection, partners: pd.DataFrame, snapshot_date) -> None:
    """つなぐ相手を ops.connect_suggestions に保存する（同じ予測時点の行は置き換える）。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.connect_suggestions ("
        "snapshot_date DATE, user_id VARCHAR, partner_id VARCHAR, similarity DOUBLE, partner_rank INTEGER)"
    )
    con.execute(
        "DELETE FROM ops.connect_suggestions WHERE snapshot_date = ?", [pd.Timestamp(snapshot_date).date()]
    )
    if len(partners):
        con.register("_partners", partners)
        con.execute("INSERT INTO ops.connect_suggestions SELECT * FROM _partners")
        con.unregister("_partners")


def save_evaluation(con: duckdb.DuckDBPyConnection, result: dict) -> None:
    """詳しい人探しの評価を ops.expertise_evaluation に追記する。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.expertise_evaluation ("
        "evaluated_at TIMESTAMP, queries INTEGER, members INTEGER, top_n INTEGER, "
        "hit_model DOUBLE, hit_activity DOUBLE, hit_random DOUBLE)"
    )
    if not result.get("queries"):
        return
    n = result["top_n"]
    con.execute(
        "INSERT INTO ops.expertise_evaluation VALUES (now(), ?, ?, ?, ?, ?, ?)",
        [
            result["queries"],
            result["members"],
            n,
            result[f"hit@{n}_model"],
            result[f"hit@{n}_activity"],
            result[f"hit@{n}_random"],
        ],
    )


def save_index(index: ExpertIndex, path: Path | str) -> None:
    """索引をファイルに保存する（API が読み込んで使う）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: Path | str) -> ExpertIndex | None:
    """保存した索引を読む。まだ無ければ None を返す。"""
    path = Path(path)
    return joblib.load(path) if path.exists() else None
