"""今週の予測: 評価の結果でモデルを選び、全期間で学習し直して、最新の予測時点の対象者を並べる。"""

from __future__ import annotations

import json
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

from slack_ml import churn

FEATURE_TEXT = {
    "tenure_days": ("参加してからの日数", "days", None),
    "posts_7d": ("直近7日の投稿", "count", "low"),
    "posts_28d": ("直近28日の投稿", "count", "low"),
    "replies_to_members_28d": ("直近28日の、他のメンバーへの返信", "count", "low"),
    "self_replies_28d": ("直近28日の、自分のスレッドへの書き足し", "count", "low"),
    "reactions_to_members_28d": ("直近28日の、他のメンバーへのリアクション", "count", "low"),
    "bot_interactions_28d": ("直近28日の、日報などボットの投稿への反応", "count", "low"),
    "actions_28d": ("直近28日の行動", "count", "low"),
    "actions_prev_28d": ("その前の28日の行動", "count", "low"),
    "activity_trend": ("行動の増減（直近28日 ÷ その前の28日）", "ratio", "low"),
    "replies_received_28d": ("直近28日に、他のメンバーからもらった返信", "count", "low"),
    "reactions_received_28d": ("直近28日に、他のメンバーからもらったリアクション", "count", "low"),
    "feedback_per_post_28d": ("投稿1回あたりにもらった反応", "ratio", "low"),
    "posts_without_reply_28d": ("直近28日の、返信が付かなかった投稿", "count", None),
    "unreplied_share_28d": ("投稿のうち、返信が付かなかった割合", "share", "high"),
    "active_days_28d": ("直近28日のうち、行動した日数", "days", "low"),
    "distinct_partners_28d": ("直近28日にやり取りしたメンバーの人数", "people", "low"),
    "channels_28d": ("直近28日に投稿したチャンネルの数", "channels", "low"),
    "days_since_last_post": ("最後の投稿からの日数", "days", "high"),
    "days_since_last_action": ("最後の行動からの日数", "days", "high"),
    "avg_text_len_28d": ("直近28日の投稿の平均文字数", "chars", "low"),
    "first_week_posts": ("参加して最初の7日の投稿", "count", "low"),
    "first_week_feedback": ("参加して最初の7日にもらった反応", "count", "low"),
}


def on_risky_side(feature: str, value: float, reference: float) -> bool:
    """値が「危ない向き」に、継続している人の中央値からずれているか。"""
    side = FEATURE_TEXT.get(feature, (feature, "ratio", None))[2]
    if side is None or pd.isna(value) or pd.isna(reference):
        return False
    return value < reference if side == "low" else value > reference


def format_value(feature: str, value: float) -> str:
    """特徴量の値を、単位付きの読みやすい文字列にする。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "なし"
    kind = FEATURE_TEXT.get(feature, (feature, "ratio", None))[1]
    if kind == "share":
        return f"{value:.0%}"
    if kind == "ratio":
        return f"{value:.2f}"
    unit = {"count": "回", "people": "人", "channels": "チャンネル", "days": "日", "chars": "文字"}[kind]
    return f"{value:,.0f}{unit}"


def select_model(summary: pd.DataFrame) -> str:
    """評価の表から、早期発見の PR-AUC が最も高いものを選ぶ（同点なら表の上の方 = 単純な方）。"""
    early = summary[summary["population"] == "early"].reset_index(drop=True)
    return str(early.loc[early["pr_auc"].idxmax(), "model"])


def load_latest(con: duckdb.DuckDBPyConnection, exclude_users=()) -> pd.DataFrame:
    """最新の予測時点の対象者を読む。exclude_users の人は対象から外す（データは学習に残す）。"""
    df = con.execute(
        """
        SELECT *
        FROM gold.feature_snapshots
        WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM gold.feature_snapshots)
        ORDER BY user_id
        """
    ).df()
    df = df[~df["user_id"].isin(set(exclude_users))].reset_index(drop=True)
    df["early"] = df["days_since_last_action"] <= churn.EARLY_DAYS
    return df


def fit_final(labeled: pd.DataFrame, name: str, random_state: int = 42):
    """選んだモデルを、ラベルのある全期間で学習し直す。ルールなら学習は要らないので None を返す。"""
    if name in churn.RULES:
        return None
    model = churn.make_models(random_state)[name]
    model.fit(labeled[churn.feature_columns(labeled)], labeled[churn.LABEL])
    return model


def predict(frame: pd.DataFrame, name: str, model, features: list[str]) -> np.ndarray:
    """リスクのスコア（大きいほど危ない）。モデルなら確率、ルールならルールの値。"""
    if model is None:
        return churn.RULES[name](frame).to_numpy(dtype=float)
    return model.predict_proba(frame[features])[:, 1]


def reasons(
    frame: pd.DataFrame, name: str, model, features: list[str], reference: pd.Series, top_n: int = 3
) -> list[list[dict]]:
    """1人ずつ、リスクを押し上げている特徴量を最大 top_n 個返す（置き換えたときの予測の下がり幅の大きい順）。"""
    if model is None:
        feature = "actions_28d" if name == "rule_low_activity" else "days_since_last_action"
        return [
            [{"feature": feature, "value": row[feature], "reference": reference[feature], "effect": None}]
            for _, row in frame.iterrows()
        ]
    base = predict(frame, name, model, features)
    effects = {}
    for feature in features:
        replaced = frame[features].copy()
        replaced[feature] = reference[feature]
        effects[feature] = base - predict(replaced, name, model, features)
    result = []
    for i, (_, row) in enumerate(frame.iterrows()):
        candidates = [f for f in features if effects[f][i] > 0 and on_risky_side(f, row[f], reference[f])]
        ranked = sorted(candidates, key=lambda f, i=i: effects[f][i], reverse=True)
        result.append(
            [
                {"feature": f, "value": row[f], "reference": reference[f], "effect": float(effects[f][i])}
                for f in ranked[:top_n]
            ]
        )
    return result


def reason_text(reason: dict) -> str:
    label = FEATURE_TEXT.get(reason["feature"], (reason["feature"], "ratio", None))[0]
    value = format_value(reason["feature"], reason["value"])
    reference = format_value(reason["feature"], reason["reference"])
    return f"{label}が {value}（継続している人の中央値は {reference}）"


def score_latest(
    labeled: pd.DataFrame, latest: pd.DataFrame, name: str, random_state: int = 42
) -> pd.DataFrame:
    """最新の予測時点の対象者を採点し、順位と理由を付けて返す。"""
    features = churn.feature_columns(labeled)
    model = fit_final(labeled, name, random_state)
    reference = labeled.loc[labeled[churn.LABEL] == 0, features].median()
    scored = latest[[*churn.KEYS, "early", "days_since_last_action", "actions_28d"]].copy()
    scored["risk"] = predict(latest, name, model, features)
    scored["reasons"] = reasons(latest, name, model, features, reference)
    scored["rank_all"] = scored["risk"].rank(ascending=False, method="first").astype(int)
    scored["rank_early"] = (
        scored["risk"].where(scored["early"]).rank(ascending=False, method="first").astype("Int64")
    )
    scored["model"] = name
    return scored.sort_values("rank_all").reset_index(drop=True)


def save_scores(con: duckdb.DuckDBPyConnection, scored: pd.DataFrame) -> int:
    """予測を ops.risk_scores に保存する。同じ予測時点とモデルの行は置き換える（何度実行しても同じ）。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.risk_scores ("
        "snapshot_date DATE, user_id VARCHAR, model VARCHAR, risk DOUBLE, rank_all INTEGER, "
        "rank_early INTEGER, early BOOLEAN, reasons VARCHAR, scored_at TIMESTAMP)"
    )
    rows = scored.assign(
        reasons=scored["reasons"].map(lambda rs: json.dumps(rs, ensure_ascii=False, default=float)),
        scored_at=datetime.now(),
    )[
        [
            "snapshot_date",
            "user_id",
            "model",
            "risk",
            "rank_all",
            "rank_early",
            "early",
            "reasons",
            "scored_at",
        ]
    ]
    for snapshot_date, model in rows[["snapshot_date", "model"]].drop_duplicates().itertuples(index=False):
        con.execute(
            "DELETE FROM ops.risk_scores WHERE snapshot_date = ? AND model = ?",
            [pd.Timestamp(snapshot_date).date(), model],
        )
    con.register("_scores", rows)
    con.execute("INSERT INTO ops.risk_scores SELECT * FROM _scores")
    con.unregister("_scores")
    return len(rows)


def save_evaluation(con: duckdb.DuckDBPyConnection, summary: pd.DataFrame, selected: str, k: int) -> None:
    """ウォークフォワード評価の表を ops.model_evaluation に追記する（いつ、どのモデルを選んだか）。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.model_evaluation ("
        "evaluated_at TIMESTAMP, k INTEGER, population VARCHAR, model VARCHAR, rows BIGINT, positives BIGINT, "
        "pr_auc DOUBLE, precision_at_k DOUBLE, recall_at_k DOUBLE, lift DOUBLE, selected BOOLEAN)"
    )
    rows = pd.DataFrame(
        {
            "evaluated_at": datetime.now(),
            "k": k,
            "population": summary["population"],
            "model": summary["model"],
            "rows": summary["rows"],
            "positives": summary["positives"],
            "pr_auc": summary["pr_auc"],
            "precision_at_k": summary[f"precision@{k}"],
            "recall_at_k": summary[f"recall@{k}"],
            "lift": summary["lift"],
            "selected": summary["model"] == selected,
        }
    )
    con.register("_evaluation", rows)
    con.execute("INSERT INTO ops.model_evaluation SELECT * FROM _evaluation")
    con.unregister("_evaluation")
