"""モデルを出した後の見守り: 特徴量の分布の変化（ドリフト）と、予測確率のずれ（キャリブレーション）。"""

from __future__ import annotations

from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

EXPECTED_GROWTH = {"tenure_days"}
EPSILON = 1e-4


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """2つの分布の PSI。区間は expected（学習データ）の分位で切る。"""
    expected = pd.to_numeric(expected, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")
    values = expected.dropna()
    if values.empty or actual.empty:
        return float("nan")
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])

    def shares(series: pd.Series) -> np.ndarray:
        counts = pd.cut(series.dropna(), edges, include_lowest=True).value_counts(sort=False).to_numpy()
        counts = np.append(counts, series.isna().sum())
        return np.maximum(counts / len(series), EPSILON)

    e, a = shares(expected), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def status(feature: str, value: float) -> str:
    if feature in EXPECTED_GROWTH:
        return "expected"
    if np.isnan(value):
        return "unknown"
    return "stable" if value < 0.1 else "warning" if value <= 0.25 else "alert"


def drift(train: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """特徴量ごとの PSI と、学習データと今週の中央値。PSI の大きい順。"""
    rows = []
    for feature in features:
        value = psi(train[feature], current[feature])
        rows.append(
            {
                "feature": feature,
                "psi": value,
                "status": status(feature, value),
                "train_median": float(train[feature].median()),
                "current_median": float(current[feature].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def calibration(
    preds: pd.DataFrame, score: str, label: str = "churned", bins: int = 10
) -> tuple[pd.DataFrame, dict]:
    """予測確率の大きさで区間に分け、区間ごとに予測の平均と実際に離脱した割合を比べる。"""
    y = preds[label].astype(float).to_numpy()
    p = preds[score].astype(float).to_numpy()
    groups = pd.qcut(pd.Series(p).rank(method="first"), q=min(bins, len(p)), labels=False).to_numpy()
    table = (
        pd.DataFrame({"group": groups + 1, "predicted": p, "observed": y})
        .groupby("group")
        .agg(
            n=("predicted", "size"), mean_predicted=("predicted", "mean"), observed_rate=("observed", "mean")
        )
        .reset_index()
    )
    rate = y.mean()
    metrics = {
        "rows": len(y),
        "positives": int(y.sum()),
        "brier": float(np.mean((p - y) ** 2)),
        "baseline_brier": float(rate * (1 - rate)),
        "ece": float(np.sum(table["n"] / len(y) * (table["mean_predicted"] - table["observed_rate"]).abs())),
    }
    return table, metrics


def save_drift(con: duckdb.DuckDBPyConnection, table: pd.DataFrame, snapshot_date) -> None:
    """ドリフトを ops.feature_drift に保存する（同じ予測時点の行は置き換える）。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.feature_drift ("
        "snapshot_date DATE, feature VARCHAR, psi DOUBLE, status VARCHAR, train_median DOUBLE, current_median DOUBLE)"
    )
    day = pd.Timestamp(snapshot_date).date()
    con.execute("DELETE FROM ops.feature_drift WHERE snapshot_date = ?", [day])
    rows = table.assign(snapshot_date=day)[
        ["snapshot_date", "feature", "psi", "status", "train_median", "current_median"]
    ]
    con.register("_drift", rows)
    con.execute("INSERT INTO ops.feature_drift SELECT * FROM _drift")
    con.unregister("_drift")


def save_calibration(
    con: duckdb.DuckDBPyConnection, table: pd.DataFrame, metrics: dict, model: str, population: str
) -> None:
    """確率のずれを ops.calibration に追記する（区間ごとに1行。指標は全行に同じ値を入れる）。"""
    con.execute(
        "CREATE TABLE IF NOT EXISTS ops.calibration ("
        "evaluated_at TIMESTAMP, model VARCHAR, population VARCHAR, grp INTEGER, n BIGINT, "
        "mean_predicted DOUBLE, observed_rate DOUBLE, brier DOUBLE, baseline_brier DOUBLE, ece DOUBLE)"
    )
    rows = pd.DataFrame(
        {
            "evaluated_at": datetime.now(),
            "model": model,
            "population": population,
            "grp": table["group"].astype(int),
            "n": table["n"].astype(int),
            "mean_predicted": table["mean_predicted"],
            "observed_rate": table["observed_rate"],
            "brier": metrics["brier"],
            "baseline_brier": metrics["baseline_brier"],
            "ece": metrics["ece"],
        }
    )
    con.register("_calibration", rows)
    con.execute("INSERT INTO ops.calibration SELECT * FROM _calibration")
    con.unregister("_calibration")
