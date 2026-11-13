"""モデルの見守り（monitoring）のテスト。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slack_ml import churn, monitoring, pipeline, report, scoring
from slack_ml import warehouse as wh


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    same = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(1, 1, 5000))
    assert monitoring.psi(base, same) < 0.02
    assert monitoring.psi(base, shifted) > 0.25


def test_psi_counts_missing_values_as_their_own_bin():
    """学習では無かった NULL が今週たくさん出たら、分布が変わったとみなす（取り込みの不具合に気付ける）。"""
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 1, 2000))
    half_missing = pd.Series(np.where(rng.random(2000) < 0.5, np.nan, rng.normal(0, 1, 2000)))
    assert monitoring.psi(base, half_missing) > 0.25


def test_psi_handles_features_with_few_distinct_values():
    """0 が大半の回数の特徴量でも、区間の境目が重ならずに計算できる。"""
    base = pd.Series([0] * 900 + [1] * 80 + [5] * 20)
    assert monitoring.psi(base, base.copy()) == pytest.approx(0, abs=1e-9)
    assert monitoring.psi(base, pd.Series([0] * 500 + [1] * 300 + [5] * 200)) > 0.25


def test_status_thresholds_and_expected_growth():
    assert monitoring.status("posts_28d", 0.05) == "stable"
    assert monitoring.status("posts_28d", 0.2) == "warning"
    assert monitoring.status("posts_28d", 0.3) == "alert"
    assert monitoring.status("tenure_days", 0.9) == "expected"
    assert monitoring.status("posts_28d", float("nan")) == "unknown"


def test_calibration_is_small_when_probabilities_are_right_and_large_when_overconfident():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 20000)
    y = rng.random(20000) < p
    _, good = monitoring.calibration(pd.DataFrame({"score": p, "churned": y}), "score")
    _, bad = monitoring.calibration(pd.DataFrame({"score": np.full(20000, 0.9), "churned": y}), "score")
    assert good["ece"] < 0.02
    assert bad["ece"] == pytest.approx(0.4, abs=0.02)
    assert good["brier"] < good["baseline_brier"]


def test_calibration_table_and_brier_on_a_small_example():
    preds = pd.DataFrame({"score": [0.1, 0.2, 0.8, 0.9], "churned": [0, 0, 1, 0]})
    table, metrics = monitoring.calibration(preds, "score", bins=2)
    assert list(table["n"]) == [2, 2]
    assert list(table["mean_predicted"].round(3)) == [0.15, 0.85]
    assert list(table["observed_rate"]) == [0.0, 0.5]
    assert metrics["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.81) / 4)
    assert metrics["baseline_brier"] == pytest.approx(0.25 * 0.75)
    assert metrics["ece"] == pytest.approx(0.5 * 0.15 + 0.5 * 0.35)


def test_drift_is_saved_once_per_snapshot_and_shown_in_the_report(export, tmp_path):
    cfg, _ = export
    pipeline.run(cfg, log=lambda *_: None)
    con = wh.connect(cfg.paths.warehouse)
    labeled = churn.load_labeled(con)
    latest = scoring.load_latest(con)
    scored = scoring.score_latest(labeled, latest, "logistic")
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

    features = churn.feature_columns(labeled)
    drift = monitoring.drift(labeled, latest, features)
    assert set(drift["feature"]) == set(features)
    assert drift.loc[drift["feature"] == "tenure_days", "status"].item() == "expected"
    snapshot = scored["snapshot_date"].iloc[0]
    monitoring.save_drift(con, drift, snapshot)
    monitoring.save_drift(con, drift, snapshot)
    [(saved,)] = con.execute("SELECT COUNT(*) FROM ops.feature_drift").fetchall()
    assert saved == len(features)

    preds = pd.DataFrame({"logistic": [0.1, 0.2, 0.8, 0.9] * 5, "churned": [0, 0, 1, 1] * 5})
    for population in ["early", "all"]:
        table, metrics = monitoring.calibration(preds, "logistic")
        monitoring.save_calibration(con, table, metrics, "logistic", population)

    path = report.build_report(con, tmp_path / "weekly.html", k)
    con.close()
    page = path.read_text(encoding="utf-8")
    assert "モデルの見守り" in page
    assert "特徴量の分布の変化（PSI）" in page
    assert "確率のずれ（logistic、早期発見）" in page
