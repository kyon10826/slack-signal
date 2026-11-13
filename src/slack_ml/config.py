"""設定の読み込み"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    export_dir: Path = Path("data/raw/slack_export")
    metadata_export: Path = Path("data/raw/metadata_export.zip")
    ground_truth_dir: Path = Path("data/raw/_ground_truth")
    warehouse: Path = Path("data/warehouse.duckdb")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42
    n_initial_members: int = 220
    weekly_new_members: float = 4.0
    start_date: date = date(2025, 11, 3)
    n_days: int = 364


@dataclass(frozen=True)
class LabelConfig:
    observation_days: int = 28
    horizon_days: int = 28
    min_tenure_days: int = 7
    snapshot_every_days: int = 7
    warmup_days: int = 56


@dataclass(frozen=True)
class ModelConfig:
    capacity_k: int = 20
    cv_folds: int = 4
    cv_fold_snapshots: int = 3
    test_snapshots: int = 8
    random_state: int = 42
    n_search_iter: int = 12
    exclude_from_scoring: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpertiseConfig:
    lookback_days: int = 120
    n_components: int = 64
    min_messages: int = 3


@dataclass(frozen=True)
class RecommendConfig:
    top_k_members: int = 20
    n_experts: int = 3
    n_channels: int = 2
    excluded_channels: tuple[str, ...] = ("general", "random", "introductions", "events")


@dataclass(frozen=True)
class PrivacyConfig:
    exclude_users: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlackConfig:
    digest_channel: str = "#community-managers"
    manager_user_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    labeling: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    expertise: ExpertiseConfig = field(default_factory=ExpertiseConfig)
    recommend: RecommendConfig = field(default_factory=RecommendConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)

    def with_root(self, root: Path) -> Config:
        """全パスを root 配下に付け替えたコピーを返す（テストや一時実行用）。"""
        p = self.paths
        new_paths = PathsConfig(**{f.name: Path(root) / getattr(p, f.name) for f in dataclasses.fields(p)})
        return dataclasses.replace(self, paths=new_paths)


def _coerce(tp: Any, value: Any) -> Any:
    """YAML の素朴な値を dataclass のフィールド型に合わせて変換する。"""
    if dataclasses.is_dataclass(tp):
        return _build(tp, value or {})
    if tp is Path:
        return Path(value)
    if tp is date:
        return value if isinstance(value, date) else date.fromisoformat(str(value))
    origin = typing.get_origin(tp)
    if origin is tuple:
        return tuple(value or ())
    if tp in (int, float, str):
        return tp(value)
    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__} に未知の設定キーがあります: {sorted(unknown)}")
    kwargs = {name: _coerce(hints[name], value) for name, value in data.items()}
    return cls(**kwargs)


REMOVED_SECTIONS = {
    "experiment": "効果検証（A/B テスト）はやめました。職場で「あえて声をかけない人」を作らないためです",
}


def load_config(path: str | Path | None = None) -> Config:
    """設定ファイルを読み込む。path が None ならデフォルト値のみを使う。"""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for section in sorted(set(data) & set(REMOVED_SECTIONS)):
        raise ValueError(f"{path} の {section}: の節を消してください。{REMOVED_SECTIONS[section]}。")
    return _build(Config, data)
