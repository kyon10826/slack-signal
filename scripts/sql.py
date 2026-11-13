"""DWH に SQL を投げて、結果を表で表示する（読み取り専用で開くので、データは変わらない）。"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config

query = sys.argv[1]
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else "configs/local.yaml")
con = wh.connect(cfg.paths.warehouse, read_only=True)
con.sql(query).show(max_rows=100)
con.close()
