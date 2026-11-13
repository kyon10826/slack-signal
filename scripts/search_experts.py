"""質問文に詳しそうなメンバーを探す（詳しい人探し）。

使い方:
    python scripts/search_experts.py "pandas で集計を速くしたい"
    python scripts/search_experts.py "pandas で集計を速くしたい" configs/default.yaml
"""

import sys
from datetime import timedelta

from slack_ml import expertise
from slack_ml import warehouse as wh
from slack_ml.config import load_config

query = sys.argv[1]
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else "configs/local.yaml")

con = wh.connect(cfg.paths.warehouse, read_only=True)
messages = expertise.load_messages(con)
names = dict(
    con.execute(
        "SELECT user_id, COALESCE(NULLIF(display_name, ''), NULLIF(real_name, ''), name, user_id) FROM silver.users"
    ).fetchall()
)
con.close()

ex = cfg.expertise
index = expertise.build_index(
    messages,
    messages["posted_date"].max() + timedelta(days=1),
    ex.lookback_days,
    ex.min_messages,
    ex.n_components,
    cfg.model.random_state,
)
if index is None:
    sys.exit("索引を作れるだけの発言がありません")

print(
    f"「{query}」に話題の近いメンバー（直近 {ex.lookback_days} 日に {ex.min_messages} 回以上発言した {len(index.user_ids)} 人から）"
)
for rank, (user_id, similarity) in enumerate(index.search(query, top_n=5), start=1):
    print(
        f"  {rank}. {names.get(user_id, user_id)}  話題の近さ {similarity:.2f}  発言 {index.message_counts[user_id]} 回"
    )
