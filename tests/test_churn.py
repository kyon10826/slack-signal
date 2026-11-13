"""離脱予兆モデルの評価の部品のテスト（小さな表で、答えが手で分かるものを確かめる）。"""

from __future__ import annotations

import pandas as pd
import pytest

from slack_ml import churn


def test_split_folds_purges_label_window():
    """学習に使う予測時点は、評価の最初の予測時点より28日以上前だけ（ラベル期間が評価期間に重ならない）。"""
    snapshots = pd.date_range("2026-01-04", periods=30, freq="7D")
    folds = churn.split_folds(snapshots, n_folds=3, fold_snapshots=4)
    assert [len(f["test"]) for f in folds] == [4, 4, 4]
    for f in folds:
        assert max(f["train"]) + pd.Timedelta(days=28) <= min(f["test"])
        assert min(f["test"]) - max(f["train"]) == pd.Timedelta(days=28)
    assert folds[0]["test"][-1] < folds[1]["test"][0] < folds[2]["test"][0]


def test_precision_recall_at_k():
    """予測時点ごとに上位 k 人を選び、全体で合計して割る。"""
    preds = pd.DataFrame(
        {
            "snapshot_date": ["a", "a", "a", "b", "b", "b"],
            "churned": [1, 0, 0, 1, 1, 0],
            "score": [0.9, 0.8, 0.1, 0.2, 0.7, 0.9],
        }
    )
    precision, recall = churn.precision_recall_at_k(preds, "score", k=1)
    assert precision == pytest.approx(1 / 2)
    assert recall == pytest.approx(1 / 3)


def test_rules_score_higher_for_riskier_members():
    """ルールのスコアは「大きいほど危ない」にそろっている。"""
    df = pd.DataFrame({"actions_28d": [1, 50], "days_since_last_action": [20, 1]})
    low_activity = churn.RULES["rule_low_activity"](df)
    recency = churn.RULES["rule_recency"](df)
    assert low_activity[0] > low_activity[1]
    assert recency[0] > recency[1]
