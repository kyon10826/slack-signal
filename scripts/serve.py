"""API を手元で起動する（http://127.0.0.1:8000/docs で、API の説明書が見られる）。

使い方:
    SLACK_ML_API_KEY=好きな文字列 python scripts/serve.py
    SLACK_ML_API_KEY=好きな文字列 python scripts/serve.py configs/default.yaml
"""

import os
import sys

import uvicorn

os.environ.setdefault("SLACK_ML_CONFIG", sys.argv[1] if len(sys.argv) > 1 else "configs/local.yaml")
uvicorn.run("slack_ml.api:app_from_env", factory=True, host="127.0.0.1", port=8000)
