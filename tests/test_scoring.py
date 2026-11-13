"""今週の予測（scoring）とレポート（report）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from slack_ml import churn, pipeline, report, scoring
from slack_ml import warehouse as wh


def score(cfg, name: str):
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse)
    labeled = churn.load_labeled(con)
    latest = scoring.load_latest(con)
    return con, scoring.score_latest(labeled, latest, name)


def test_excluded_users_are_not_scored_but_stay_in_training(export):
    """model.exclude_from_scoring の人は予測の一覧に出ないが、学習データには残る。"""
    cfg, _ = export
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse)
    labeled = churn.load_labeled(con)
    everyone = scoring.load_latest(con)
    latest = scoring.load_latest(con, exclude_users=("U1",))
    con.close()
    scored = scoring.score_latest(labeled, latest, "logistic")
    assert "U1" in set(everyone["user_id"])
    assert "U1" not in set(scored["user_id"])
    assert "U1" in set(labeled["user_id"])
    assert len(scored) == len(everyone) - 1
    assert sorted(scored["rank_all"]) == list(range(1, len(scored) + 1))


def test_select_model_uses_early_pr_auc_and_prefers_simpler_on_tie():
    models = ["rule_low_activity", "rule_recency", "logistic", "gbdt"]
    summary = pd.DataFrame(
        {
            "population": ["early"] * 4 + ["all"] * 4,
            "model": models * 2,
            "pr_auc": [0.30, 0.10, 0.30, 0.20, 0.90, 0.90, 0.90, 0.95],
        }
    )
    assert scoring.select_model(summary) == "rule_low_activity"
    summary.loc[3, "pr_auc"] = 0.50
    assert scoring.select_model(summary) == "gbdt"


def test_reasons_show_only_risky_direction():
    """モデルが直感と逆向きの関係を学んでいても、その特徴量は理由に出さない。"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {"actions_28d": rng.integers(0, 50, 500), "first_week_feedback": rng.integers(0, 20, 500)}
    )
    logit = -0.15 * X["actions_28d"] + 0.3 * X["first_week_feedback"] - 1
    y = (rng.random(500) < 1 / (1 + np.exp(-logit))).astype(int)
    model = LogisticRegression().fit(X, y)
    reference = X[y == 0].median()
    member = pd.DataFrame({"actions_28d": [1], "first_week_feedback": [19]})
    [reasons] = scoring.reasons(member, "logistic", model, list(X.columns), reference)
    assert [r["feature"] for r in reasons] == ["actions_28d"]
    assert "継続している人の中央値" in scoring.reason_text(reasons[0])


def test_scores_have_ranks_and_saving_twice_keeps_one_row_per_member(export):
    cfg, _ = export
    con, scored = score(cfg, "logistic")
    assert sorted(scored["rank_all"]) == list(range(1, len(scored) + 1))
    assert scored.loc[~scored["early"], "rank_early"].isna().all()
    assert scored["risk"].between(0, 1).all()
    scoring.save_scores(con, scored)
    scoring.save_scores(con, scored)
    [(rows, distinct)] = con.execute(
        "SELECT COUNT(*), (SELECT COUNT(*) FROM (SELECT DISTINCT snapshot_date, user_id, model FROM ops.risk_scores)) "
        "FROM ops.risk_scores"
    ).fetchall()
    con.close()
    assert rows == distinct == len(scored)


def test_rule_scores_use_the_rule_feature(export):
    cfg, _ = export
    con, scored = score(cfg, "rule_low_activity")
    con.close()
    assert (scored["risk"] == -scored["actions_28d"]).all()
    assert all(r[0]["feature"] == "actions_28d" for r in scored["reasons"])


def test_report_escapes_names_and_has_all_sections(export, tmp_path):
    """メンバーの名前は HTML に埋め込む前にエスケープする（名前に <script> が入っていても実行されない）。"""
    cfg, _ = export
    con, scored = score(cfg, "logistic")
    scoring.save_scores(con, scored)
    k = 5
    summary = pd.DataFrame(
        {
            "population": ["early", "all"],
            "model": ["logistic", "logistic"],
            "rows": [10, 20],
            "positives": [1, 2],
            "pr_auc": [0.5, 0.6],
            f"precision@{k}": [0.2, 0.4],
            f"recall@{k}": [1.0, 1.0],
            "lift": [2.0, 4.0],
        }
    )
    scoring.save_evaluation(con, summary, "logistic", k)
    con.execute("UPDATE silver.users SET display_name = '<script>alert(1)</script>' WHERE user_id = 'U1'")
    path = report.build_report(con, tmp_path / "weekly.html", k, "比較の説明")
    con.close()

    page = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    for title in ["今週の数字", "今週声をかけたいメンバー", "週次の推移", "モデルの評価", "使い方の注意"]:
        assert title in page
