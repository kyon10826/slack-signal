"""bronze 層から silver 層を作る（src/slack_ml/sql/silver/ の SQL を名前順に実行する）。

使い方:
    python scripts/build_silver.py
    python scripts/build_silver.py configs/default.yaml
"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")

con = wh.connect(cfg.paths.warehouse)
for name in wh.run_sql_layer(con, "silver"):
    print(f"実行: silver/{name}")
counts = wh.table_counts(con, "silver")
con.close()

print("\nsilver 層の行数:")
for name, n in counts.items():
    print(f"  silver.{name:<14} {n:>8,}")
