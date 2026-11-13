"""毎週の処理（パイプライン、モデルの評価と選択、今週の予測、詳しい人探し、見守り、レポート）を実行する。

使い方:
    python scripts/weekly.py
    python scripts/weekly.py configs/default.yaml
"""

import sys
from datetime import timedelta

import pandas as pd

from slack_ml import churn, expertise, monitoring, pipeline, report, scoring
from slack_ml import warehouse as wh
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
k = cfg.model.capacity_k

try:
    pipeline.run(cfg)
except pipeline.QualityError as e:
    print(f"\n止まりました（予測もレポートも作りません）: {e}")
    sys.exit(1)

con = wh.connect(cfg.paths.warehouse)
labeled = churn.load_labeled(con)
preds = churn.walk_forward(labeled, cfg.model.cv_folds, cfg.model.cv_fold_snapshots, cfg.model.random_state)
summary = churn.summarize(preds, [*churn.RULES, "logistic", "gbdt"], k)
selected = scoring.select_model(summary)

early_pr_auc = summary[summary["population"] == "early"].set_index("model")["pr_auc"]
best_rule = max(churn.RULES, key=early_pr_auc.get)
if selected in churn.RULES:
    comparison = f"早期発見では、ルール {selected} の PR-AUC が最も高かったので、ルールで並べています。"
else:
    point, low, high = churn.bootstrap_pr_auc_diff(preds[preds["early"]], selected, best_rule)
    comparison = (
        f"早期発見の PR-AUC の差（{selected} − {best_rule}）は {point:+.3f}、"
        f"95%信頼区間は {low:+.3f}〜{high:+.3f} です（予測時点を単位にしたブートストラップ）。"
    )

scoring.save_evaluation(con, summary, selected, k)
latest = scoring.load_latest(con, cfg.model.exclude_from_scoring)
scored = scoring.score_latest(labeled, latest, selected, cfg.model.random_state)
saved = scoring.save_scores(con, scored)

messages = expertise.load_messages(con)
ex = cfg.expertise
index = None
expert_eval = {"queries": 0}
if len(messages):
    until = messages["posted_date"].max() + timedelta(days=1)
    index = expertise.build_index(
        messages, until, ex.lookback_days, ex.min_messages, ex.n_components, cfg.model.random_state
    )
    expert_eval = expertise.evaluate(
        messages,
        lookback_days=ex.lookback_days,
        min_messages=ex.min_messages,
        n_components=ex.n_components,
        random_state=cfg.model.random_state,
    )
expertise.save_evaluation(con, expert_eval)
partners = expertise.suggest_partners(index, scored, k, cfg.recommend.n_experts)
expertise.save_partners(con, partners, scored["snapshot_date"].iloc[0])
if index is not None:
    expertise.save_index(index, cfg.paths.artifacts_dir / "expertise.joblib")

drift = monitoring.drift(labeled, latest, churn.feature_columns(labeled))
monitoring.save_drift(con, drift, scored["snapshot_date"].iloc[0])
calibration = {}
if selected not in churn.RULES:
    for population, rows in [("early", preds[preds["early"]]), ("all", preds)]:
        table, metrics = monitoring.calibration(rows, selected)
        monitoring.save_calibration(con, table, metrics, selected, population)
        calibration[population] = metrics
path = report.build_report(
    con,
    cfg.paths.reports_dir / "weekly.html",
    k,
    comparison,
)
con.close()

pd.set_option("display.width", 200)
print(f"\n== モデルの評価（K = {k}） ==")
print(summary.round(3).to_string(index=False))
print(f"\n採用: {selected}")
print(comparison)

snapshot = pd.Timestamp(scored["snapshot_date"].iloc[0]).date()
print(
    f"\n== 今週（{snapshot}）声をかけたいメンバー: 直近7日に行動した {int(scored['early'].sum())} 人の上位 {k} 人 =="
)
if cfg.model.exclude_from_scoring:
    print(f"  予測の対象から外した人: {len(cfg.model.exclude_from_scoring)} 人（model.exclude_from_scoring）")
top = scored[scored["early"]].sort_values("rank_early").head(k)
for row in top.itertuples(index=False):
    print(f"  {row.rank_early}位  リスク {row.risk:.3f}")
    for reason in row.reasons:
        print(f"        - {scoring.reason_text(reason)}")
print("\n== 詳しい人探しの評価（返信した人が上位5人に入った割合） ==")
if expert_eval.get("queries"):
    print(
        f"  話題の近さ {expert_eval['hit@5_model']:.1%}  発言の多い順 {expert_eval['hit@5_activity']:.1%}"
        f"  ランダム {expert_eval['hit@5_random']:.1%}  （{expert_eval['queries']} 件の投稿で評価）"
    )
else:
    print("  評価に使える投稿がありませんでした")
print(f"  つなぐ相手: {partners['user_id'].nunique()} 人に、のべ {len(partners)} 人を選びました")
print("\n== モデルの見守り ==")
states = drift["status"].value_counts()
print(
    f"  特徴量の分布（PSI）: 大きく変化 {states.get('alert', 0)}個、注意 {states.get('warning', 0)}個、"
    f"安定 {states.get('stable', 0)}個、時間とともに増える {states.get('expected', 0)}個"
)
for row in drift[drift["status"] == "alert"].head(5).itertuples(index=False):
    print(f"    {row.feature:<28} PSI {row.psi:.3f}  中央値 {row.train_median:g} → {row.current_median:g}")
if calibration:
    for population, m in calibration.items():
        print(
            f"  確率のずれ（{'早期発見' if population == 'early' else '全員'}）: ECE {m['ece']:.3f}、"
            f"Brier {m['brier']:.3f}（全員に平均の離脱率を言ったとき {m['baseline_brier']:.3f}）"
        )
else:
    print("  確率のずれ: ルールは確率を出さないので測りません")
print(f"\n予測 {saved} 人分を ops.risk_scores に保存しました")
print(f"レポート: {path}")
