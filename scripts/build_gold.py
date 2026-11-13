"""silver 層から gold 層を作る（src/slack_ml/sql/gold/ の SQL を名前順に実行する）。"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")

con = wh.connect(cfg.paths.warehouse)
for name in wh.run_sql_layer(con, "gold"):
    print(f"実行: gold/{name}")
counts = wh.table_counts(con, "gold")
con.close()

print("\ngold 層の行数:")
for name, n in counts.items():
    print(f"  gold.{name:<14} {n:>8,}")
