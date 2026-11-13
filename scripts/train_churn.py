"""離脱予兆モデルを、時系列で分けた評価（ウォークフォワード）でルールと比べる。

使い方:
    python scripts/train_churn.py
    python scripts/train_churn.py configs/default.yaml
"""

import sys

import pandas as pd

from slack_ml import churn
from slack_ml import warehouse as wh
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
k = cfg.model.capacity_k

con = wh.connect(cfg.paths.warehouse, read_only=True)
df = churn.load_labeled(con)
con.close()

folds = churn.split_folds(df["snapshot_date"], cfg.model.cv_folds, cfg.model.cv_fold_snapshots)
print(f"ラベルのある行: {len(df):,}（陽性 {int(df['churned'].sum())}）")
for f in folds:
    print(
        f"  fold {f['fold']}: 学習 {f['train'][0].date()}〜{f['train'][-1].date()}（{len(f['train'])}回）"
        f" → 評価 {f['test'][0].date()}〜{f['test'][-1].date()}（{len(f['test'])}回）"
    )

preds = churn.walk_forward(df, cfg.model.cv_folds, cfg.model.cv_fold_snapshots, cfg.model.random_state)
scores = [*churn.RULES, "logistic", "gbdt"]

pd.set_option("display.width", 200)
print(f"\n== 評価期間の成績（K = {k}） ==")
print(churn.summarize(preds, scores, k).round(3).to_string(index=False))

print("\n== PR-AUC の差と95%信頼区間（予測時点を単位にしたブートストラップ） ==")
for population, part in [("early", preds[preds["early"]]), ("all", preds)]:
    for model in ["logistic", "gbdt"]:
        for rule in churn.RULES:
            point, low, high = churn.bootstrap_pr_auc_diff(part, model, rule)
            print(f"  {population:<5} {model:<8} − {rule:<17} {point:+.3f} [{low:+.3f}, {high:+.3f}]")

print("\n== 特徴量の重要さ（最後の fold で、並べ替えたときの PR-AUC の下がり幅。勾配ブースティング） ==")
importance = churn.permutation_importance_last_fold(
    df, cfg.model.cv_folds, cfg.model.cv_fold_snapshots, cfg.model.random_state
)
print(importance.round(4).to_string())
