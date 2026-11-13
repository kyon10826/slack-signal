"""離脱予兆モデル: 時系列で分けた評価（ウォークフォワード）で、ルールと機械学習を比べる。"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

KEYS = ["snapshot_date", "user_id"]
LABEL = "churned"
HORIZON_DAYS = 28
EARLY_DAYS = 7

RULES = {
    "rule_low_activity": lambda df: -df["actions_28d"],
    "rule_recency": lambda df: df["days_since_last_action"],
}


def load_labeled(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """ラベルのある行だけを読み、早期発見の対象かどうか（early）の列を足す。"""
    df = con.execute(
        "SELECT * FROM gold.feature_snapshots WHERE churned IS NOT NULL ORDER BY snapshot_date, user_id"
    ).df()
    df["early"] = df["days_since_last_action"] <= EARLY_DAYS
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [*KEYS, LABEL, "early"]]


def make_models(random_state: int = 42) -> dict:
    """比べる2つのモデル。どちらも、ハイパーパラメータは調整せず、決めた値で比べる。"""
    return {
        "logistic": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=2000),
        ),
        "gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def split_folds(snapshots, n_folds: int, fold_snapshots: int, horizon_days: int = HORIZON_DAYS) -> list[dict]:
    """予測時点の列を、ウォークフォワードの (学習, 評価) の組に分ける。"""
    snaps = sorted(pd.to_datetime(pd.Series(snapshots)).unique())
    evaluation = snaps[-n_folds * fold_snapshots :]
    folds = []
    for k in range(n_folds):
        test = evaluation[k * fold_snapshots : (k + 1) * fold_snapshots]
        train = [s for s in snaps if s + pd.Timedelta(days=horizon_days) <= test[0]]
        folds.append({"fold": k, "train": train, "test": test})
    return folds


def walk_forward(df: pd.DataFrame, n_folds: int, fold_snapshots: int, random_state: int = 42) -> pd.DataFrame:
    """各 fold で学習し直して評価期間を予測する。返すのは、評価期間の全行のスコア（ルールとモデル）。"""
    features = feature_columns(df)
    dates = pd.to_datetime(df["snapshot_date"])
    outputs = []
    for fold in split_folds(df["snapshot_date"], n_folds, fold_snapshots):
        train = df[dates.isin(fold["train"])]
        test = df[dates.isin(fold["test"])]
        out = test[[*KEYS, LABEL, "early"]].copy()
        out["fold"] = fold["fold"]
        for name, rule in RULES.items():
            out[name] = rule(test).to_numpy()
        for name, model in make_models(random_state).items():
            model.fit(train[features], train[LABEL])
            out[name] = model.predict_proba(test[features])[:, 1]
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def precision_recall_at_k(preds: pd.DataFrame, score: str, k: int) -> tuple[float, float]:
    """予測時点ごとに上位 k 人を選び、全予測時点で合計した適合率と再現率を返す。"""
    hits = picked = positives = 0
    for _, group in preds.groupby("snapshot_date"):
        top = group.sort_values(score, ascending=False, kind="mergesort").head(k)
        hits += int(top[LABEL].sum())
        picked += len(top)
        positives += int(group[LABEL].sum())
    return hits / picked, (hits / positives if positives else float("nan"))


def summarize(preds: pd.DataFrame, scores: list[str], k: int) -> pd.DataFrame:
    rows = []
    for population, part in [("early", preds[preds["early"]]), ("all", preds)]:
        base_rate = part[LABEL].mean()
        for score in scores:
            precision, recall = precision_recall_at_k(part, score, k)
            rows.append(
                {
                    "population": population,
                    "model": score,
                    "rows": len(part),
                    "positives": int(part[LABEL].sum()),
                    "pr_auc": average_precision_score(part[LABEL], part[score]),
                    f"precision@{k}": precision,
                    f"recall@{k}": recall,
                    "lift": precision / base_rate,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_pr_auc_diff(
    preds: pd.DataFrame, a: str, b: str, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """PR-AUC の差（a − b）と、予測時点を単位に復元抽出した95%信頼区間を返す。"""
    y = preds[LABEL].to_numpy()
    sa, sb = preds[a].to_numpy(), preds[b].to_numpy()
    groups = list(preds.groupby("snapshot_date").indices.values())
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], sa[idx]) - average_precision_score(y[idx], sb[idx]))
    point = average_precision_score(y, sa) - average_precision_score(y, sb)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return point, low, high


def permutation_importance_last_fold(
    df: pd.DataFrame, n_folds: int, fold_snapshots: int, random_state: int = 42
) -> pd.Series:
    """最後の fold で、特徴量を1つずつ並べ替えたときに PR-AUC がどれだけ下がるか（大きいほど効いている）。"""
    features = feature_columns(df)
    dates = pd.to_datetime(df["snapshot_date"])
    last = split_folds(df["snapshot_date"], n_folds, fold_snapshots)[-1]
    train, test = df[dates.isin(last["train"])], df[dates.isin(last["test"])]
    model = make_models(random_state)["gbdt"].fit(train[features], train[LABEL])
    result = permutation_importance(
        model,
        test[features],
        test[LABEL],
        scoring="average_precision",
        n_repeats=5,
        random_state=random_state,
    )
    return pd.Series(result.importances_mean, index=features).sort_values(ascending=False)
