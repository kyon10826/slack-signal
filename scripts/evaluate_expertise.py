"""詳しい人探しを、「投稿に返信した人が上位 N 人に入るか」で評価する（件数と割合だけを表示する）。

使い方:
    python scripts/evaluate_expertise.py
    python scripts/evaluate_expertise.py configs/default.yaml
"""

import sys

from slack_ml import expertise
from slack_ml import warehouse as wh
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
messages = expertise.load_messages(con)
con.close()

ex = cfg.expertise
for top_n in (3, 5, 10):
    result = expertise.evaluate(
        messages,
        top_n=top_n,
        lookback_days=ex.lookback_days,
        min_messages=ex.min_messages,
        n_components=ex.n_components,
        random_state=cfg.model.random_state,
    )
    if not result.get("queries"):
        sys.exit("評価に使える投稿がありません")
    if top_n == 3:
        print(
            f"評価期間 {result['eval_start']} から、{result['queries']} 件の投稿を質問文にし、"
            f"{result['members']} 人の中から検索しました"
        )
    print(
        f"  上位 {top_n:>2} 人に返信した人が入った割合: 話題の近さ {result[f'hit@{top_n}_model']:.1%}"
        f"  発言の多い順 {result[f'hit@{top_n}_activity']:.1%}  ランダム {result[f'hit@{top_n}_random']:.1%}"
    )
