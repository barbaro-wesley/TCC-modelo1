"""Verified, causal consumer for DieselNews weekly-signal.v3 snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .selection import S10Supervised

_POINTER_FIELDS = {
    "dataset_sha256",
    "pointer_sha256",
    "pointer_version",
    "relative_path",
    "schema_version",
}
_MANIFEST_FIELDS = {
    "config_sha256",
    "dataset_sha256",
    "dataset_type",
    "files",
    "generated_through_forecast_at",
    "manifest_version",
    "parent_news_dataset_sha256",
    "pipeline_version",
    "record_count",
    "schema_version",
    "source_catalog_sha256",
}
_SIGNAL_FIELDS = {
    "active_source_count",
    "article_count",
    "category_counts",
    "direction_counts",
    "exact_duplicate_suppressed_count",
    "forecast_at",
    "lookback_days",
    "mean_confidence",
    "mean_intensity",
    "mean_relevance_s10",
    "missing_summary_count",
    "no_news",
    "observed_source_count",
    "revision_superseded_count",
    "schema_version",
    "signal_id",
    "source_coverage",
    "target_week_start",
    "unavailable_count",
    "week_convention",
}
_DIRECTIONS = ("down", "neutral", "uncertain", "up")
_CATEGORIES = (
    "biodiesel",
    "demand",
    "logistics_strike",
    "oil_brent",
    "refining_petrobras",
    "regulation",
    "supply",
    "tax",
    "usd_brl",
)

NEWS_CORE_FEATURES = (
    "news_article_count_log1p",
    "news_source_coverage",
    "news_relevance",
    "news_intensity",
    "news_confidence",
    "news_direction_balance",
    "news_no_news",
)
NEWS_ALL_FEATURES = (
    *NEWS_CORE_FEATURES,
    "news_directional_share",
    "news_uncertain_share",
    "news_missing_summary_share",
    "news_exact_duplicate_suppressed_log1p",
    *(f"news_category_{category}_share" for category in _CATEGORIES),
)


@dataclass(frozen=True)
class NewsFeatureSnapshot:
    """Verified feature frame and immutable upstream lineage."""

    frame: pd.DataFrame
    dataset_sha256: str
    parent_news_dataset_sha256: str
    source_catalog_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(data: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _regular_file(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is not a regular file: {path}")
    return path.read_bytes()


def _resolve_snapshot(path: Path) -> tuple[Path, dict[str, Any]]:
    if (path / "manifest.json").is_file():
        snapshot = path
    else:
        pointer_path = path if path.name == "latest.json" else path / "latest.json"
        pointer = _json_object(
            _regular_file(pointer_path, context="news latest pointer"),
            context="news latest pointer",
        )
        if set(pointer) != _POINTER_FIELDS:
            raise ValueError("news latest pointer fields differ from the v1 contract")
        if pointer.get("pointer_version") != "dataset-pointer.v1":
            raise ValueError("unsupported news pointer version")
        if pointer.get("schema_version") != "weekly-signal.v3":
            raise ValueError("news pointer schema is not weekly-signal.v3")
        dataset_hash = _hash(pointer.get("dataset_sha256"), field="dataset_sha256")
        pointer_hash = _hash(pointer.get("pointer_sha256"), field="pointer_sha256")
        core = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
        if _digest(_canonical_json(core)) != pointer_hash:
            raise ValueError("news pointer checksum mismatch")
        relative_path = f"snapshots/{dataset_hash}"
        if pointer.get("relative_path") != relative_path:
            raise ValueError("news pointer relative path is not canonical")
        snapshot = pointer_path.parent / "snapshots" / dataset_hash
    manifest = _json_object(
        _regular_file(snapshot / "manifest.json", context="news manifest"),
        context="news manifest",
    )
    return snapshot, manifest


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _unit_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return number


def _count_map(value: object, *, field: str, keys: tuple[str, ...]) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{field} keys differ from the weekly-signal.v3 contract")
    return {
        key: _nonnegative_integer(value[key], field=f"{field}.{key}") for key in keys
    }


def _feature_row(value: dict[str, Any], *, horizon_weeks: int) -> dict[str, object] | None:
    if set(value) != _SIGNAL_FIELDS:
        raise ValueError("weekly news signal fields differ from the v3 contract")
    if value.get("schema_version") != "weekly-signal.v3":
        raise ValueError("unexpected weekly news schema")
    if value.get("week_convention") != "anp_sunday":
        raise ValueError("weekly news signal is not aligned to the ANP Sunday calendar")
    signal_id = value.get("signal_id")
    if not isinstance(signal_id, str) or not signal_id.startswith("weekly3:") or len(signal_id) != 72:
        raise ValueError("invalid weekly news signal_id")
    forecast = pd.to_datetime(value.get("forecast_at"), utc=True, errors="raise")
    target = pd.to_datetime(value.get("target_week_start"), utc=True, errors="raise")
    if forecast.weekday() != 6 or target.weekday() != 6 or forecast != forecast.normalize():
        raise ValueError("weekly news dates must be Sunday-aligned")
    observed_horizon = (target.normalize() - forecast.normalize()).days // 7
    if target <= forecast or observed_horizon * 7 != (target - forecast).days:
        raise ValueError("weekly news target is not an exact positive-week horizon")
    if observed_horizon != horizon_weeks:
        return None

    article_count = _nonnegative_integer(value.get("article_count"), field="article_count")
    active_sources = _nonnegative_integer(
        value.get("active_source_count"), field="active_source_count"
    )
    observed_sources = _nonnegative_integer(
        value.get("observed_source_count"), field="observed_source_count"
    )
    if active_sources < 1 or observed_sources > active_sources:
        raise ValueError("invalid active/observed news source counts")
    directions = _count_map(
        value.get("direction_counts"), field="direction_counts", keys=_DIRECTIONS
    )
    categories = _count_map(
        value.get("category_counts"), field="category_counts", keys=_CATEGORIES
    )
    if sum(directions.values()) != article_count:
        raise ValueError("news direction counts do not equal article_count")
    no_news = value.get("no_news")
    if not isinstance(no_news, bool) or no_news != (article_count == 0):
        raise ValueError("no_news is inconsistent with article_count")
    source_coverage = _unit_float(value.get("source_coverage"), field="source_coverage")
    if not math.isclose(source_coverage, observed_sources / active_sources, abs_tol=1e-8):
        raise ValueError("source_coverage is inconsistent with source counts")
    denominator = max(article_count, 1)
    row: dict[str, object] = {
        "date": forecast.tz_convert(None).normalize(),
        "news_article_count_log1p": math.log1p(article_count),
        "news_source_coverage": source_coverage,
        "news_relevance": _unit_float(value.get("mean_relevance_s10"), field="mean_relevance_s10"),
        "news_intensity": _unit_float(value.get("mean_intensity"), field="mean_intensity"),
        "news_confidence": _unit_float(value.get("mean_confidence"), field="mean_confidence"),
        "news_direction_balance": (directions["up"] - directions["down"]) / denominator,
        "news_no_news": float(no_news),
        "news_directional_share": (directions["up"] + directions["down"]) / denominator,
        "news_uncertain_share": directions["uncertain"] / denominator,
        "news_missing_summary_share": _nonnegative_integer(
            value.get("missing_summary_count"), field="missing_summary_count"
        )
        / denominator,
        "news_exact_duplicate_suppressed_log1p": math.log1p(
            _nonnegative_integer(
                value.get("exact_duplicate_suppressed_count"),
                field="exact_duplicate_suppressed_count",
            )
        ),
    }
    for category in _CATEGORIES:
        row[f"news_category_{category}_share"] = categories[category] / denominator
    return row


def load_weekly_news_features(path: Path, *, horizon_weeks: int = 1) -> NewsFeatureSnapshot:
    """Verify a content-addressed v3 snapshot and return one causal row per origin."""

    if horizon_weeks < 1:
        raise ValueError("horizon_weeks must be positive")
    snapshot, manifest = _resolve_snapshot(path)
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("news manifest fields differ from the v3 contract")
    expected_versions = (
        manifest.get("manifest_version") == "dataset-manifest.v3"
        and manifest.get("dataset_type") == "weekly"
        and manifest.get("schema_version") == "weekly-signal.v3"
        and manifest.get("pipeline_version") == "temporal-impact-aggregation.v2"
    )
    if not expected_versions:
        raise ValueError("unsupported news manifest version/type/schema/pipeline")
    dataset_hash = _hash(manifest.get("dataset_sha256"), field="dataset_sha256")
    parent_hash = _hash(
        manifest.get("parent_news_dataset_sha256"),
        field="parent_news_dataset_sha256",
    )
    catalog_hash = _hash(
        manifest.get("source_catalog_sha256"), field="source_catalog_sha256"
    )
    _hash(manifest.get("config_sha256"), field="config_sha256")
    core = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if _digest(_canonical_json(core)) != dataset_hash or snapshot.name != dataset_hash:
        raise ValueError("news manifest content address mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"signals.jsonl"}:
        raise ValueError("news manifest must declare only signals.jsonl")
    expected_member_hash = _hash(files.get("signals.jsonl"), field="files.signals.jsonl")
    member = _regular_file(snapshot / "signals.jsonl", context="news signals")
    if _digest(member) != expected_member_hash:
        raise ValueError("news signals SHA-256 mismatch")
    if {item.name for item in snapshot.iterdir()} != {"manifest.json", "signals.jsonl"}:
        raise ValueError("news snapshot contains undeclared files")

    rows: list[dict[str, object]] = []
    total_records = 0
    seen_pairs: set[tuple[str, str]] = set()
    for line_number, line in enumerate(member.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank news JSONL record at line {line_number}")
        value = _json_object(line, context=f"news signal line {line_number}")
        pair = (str(value.get("forecast_at")), str(value.get("target_week_start")))
        if pair in seen_pairs:
            raise ValueError("duplicate news forecast/target pair")
        seen_pairs.add(pair)
        total_records += 1
        row = _feature_row(value, horizon_weeks=horizon_weeks)
        if row is not None:
            rows.append(row)
    if total_records != _nonnegative_integer(manifest.get("record_count"), field="record_count"):
        raise ValueError("news manifest record_count mismatch")
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"news snapshot has no horizon={horizon_weeks} rows")
    frame = frame.sort_values("date", ignore_index=True)
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("news features are not strictly ordered and unique by origin")
    return NewsFeatureSnapshot(
        frame=frame,
        dataset_sha256=dataset_hash,
        parent_news_dataset_sha256=parent_hash,
        source_catalog_sha256=catalog_hash,
    )


def augment_supervised_with_news(
    base: S10Supervised,
    snapshot: NewsFeatureSnapshot,
    *,
    feature_names: tuple[str, ...] = NEWS_CORE_FEATURES,
) -> S10Supervised:
    """Append news known at each origin without changing targets or fold boundaries."""

    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("news feature_names must be non-empty and unique")
    missing_columns = [name for name in feature_names if name not in snapshot.frame]
    if missing_columns:
        raise ValueError(f"news snapshot is missing features: {missing_columns}")
    news = snapshot.frame.set_index("date")
    origins = pd.DatetimeIndex(base.dates).normalize()
    missing_dates = origins.difference(news.index)
    if len(missing_dates):
        raise ValueError(
            f"news snapshot is missing {len(missing_dates)} supervised origins; "
            f"first={missing_dates[0].date()}"
        )
    news_values = news.loc[origins, list(feature_names)].to_numpy(dtype=float)
    if not np.all(np.isfinite(news_values)):
        raise ValueError("news feature matrix contains NaN or infinity")
    return S10Supervised(
        x=np.column_stack((base.x, news_values)),
        target_price=base.target_price.copy(),
        origin_price=base.origin_price.copy(),
        dates=base.dates.copy(),
        target_dates=base.target_dates.copy(),
        feature_names=(*base.feature_names, *feature_names),
        horizon=base.horizon,
    )
