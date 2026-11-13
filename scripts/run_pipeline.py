import sys

from slack_ml import pipeline
from slack_ml.config import load_config

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
try:
    pipeline.run(cfg)
except pipeline.QualityError as e:
    print(f"\n止まりました: {e}")
    sys.exit(1)
print("\n完了しました")
