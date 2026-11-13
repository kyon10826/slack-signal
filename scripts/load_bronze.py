"""エクスポートを読み込み、DuckDB の bronze 層に入れる。"""

import sys

from slack_ml import warehouse as wh
from slack_ml.config import load_config
from slack_ml.ingest.slack_export import read_export

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")

tables = read_export(
    cfg.paths.export_dir,
    metadata=cfg.paths.metadata_export,
    exclude_users=cfg.privacy.exclude_users,
)
print("取り込みのメモ:")
for note in tables.notes:
    print(f"  - {note}")

con = wh.connect(cfg.paths.warehouse)
counts = wh.load_bronze(con, tables)
con.close()

print(f"\n{cfg.paths.warehouse} の bronze 層に入れた行数:")
for name, n in counts.items():
    print(f"  bronze.{name:<10} {n:>8,}")
