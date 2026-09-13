"""Deterministic, double-annotation workflow for S10 news-pressure labels."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .news_pressure import NewsCorpusSnapshot

ANNOTATION_COLUMNS = (
    "batch_id",
    "news_dataset_sha256",
    "item_id",
    "annotation_slot",
    "record_id",
    "source_id",
    "published_date",
    "first_available_at",
    "canonical_url",
    "license_name",
    "title",
    "summary",
    "annotator_id",
    "annotated_at_utc",
    "relevance_label",
    "direction_label",
    "intensity_label",
    "horizon_label",
    "evidence_text",
    "rationale",
)
_DIRECTIONS = {"down", "neutral", "up", "uncertain"}
_HORIZONS = {"1w", "2w", "4w", "long", "unknown"}


@dataclass(frozen=True)
class AnnotationBatch:
    """An immutable queue plus its deterministic selection metadata."""

    frame: pd.DataFrame
    batch_id: str
    news_dataset_sha256: str
    item_count: int
    rows_per_item: int
    selection_pool_counts: dict[str, int]
    source_counts: dict[str, int]


def _stable_score(value: str, *, random_state: int) -> str:
    return hashlib.sha256(f"{random_state}:{value}".encode()).hexdigest()


def _stratified_sample(
    frame: pd.DataFrame,
    count: int,
    *,
    random_state: int,
) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    if count >= len(frame):
        return frame.copy()
    working = frame.copy()
    working["_year"] = pd.to_datetime(working["published_date"], errors="raise").dt.year
    working["_score"] = working["record_id"].map(
        lambda value: _stable_score(str(value), random_state=random_state)
    )
    group_sizes = working.groupby(["source_id", "_year"], observed=True).size()
    raw_quotas = group_sizes * (count / len(working))
    quotas = np.floor(raw_quotas).astype(int)
    remaining = count - int(quotas.sum())
    remainder_order = (raw_quotas - quotas).sort_values(ascending=False).index
    for key in remainder_order[:remaining]:
        quotas.loc[key] += 1
    selected: list[pd.DataFrame] = []
    for key, quota in quotas.items():
        if quota:
            group = working[
                (working["source_id"] == key[0]) & (working["_year"] == key[1])
            ]
            selected.append(group.sort_values("_score").head(int(quota)))
    output = pd.concat(selected, ignore_index=True)
    return output.drop(columns=["_year", "_score"])


def build_annotation_batch(
    snapshot: NewsCorpusSnapshot,
    *,
    item_count: int = 300,
    relevant_fraction: float = 2 / 3,
    annotators_per_item: int = 2,
    random_state: int = 20260823,
    batch_name: str = "s10-pressure-v1",
) -> AnnotationBatch:
    """Create a reproducible mix of lexicon-positive and control articles."""

    if item_count < 10:
        raise ValueError("item_count must be at least 10")
    if not 0.0 < relevant_fraction < 1.0:
        raise ValueError("relevant_fraction must be in (0, 1)")
    if annotators_per_item < 2:
        raise ValueError("annotators_per_item must be at least 2")
    articles = snapshot.articles.copy()
    if item_count > len(articles):
        raise ValueError("item_count exceeds the verified news corpus")
    target_relevant = round(item_count * relevant_fraction)
    relevant_pool = articles[articles["relevance_s10"] > 0.0]
    control_pool = articles[articles["relevance_s10"] == 0.0]
    relevant_count = min(target_relevant, len(relevant_pool))
    control_count = min(item_count - relevant_count, len(control_pool))
    if relevant_count + control_count < item_count:
        relevant_count = min(len(relevant_pool), item_count - control_count)
    if relevant_count + control_count != item_count:
        raise ValueError("news corpus cannot satisfy the requested annotation mix")
    relevant = _stratified_sample(
        relevant_pool, relevant_count, random_state=random_state
    ).assign(selection_pool="lexicon_relevant")
    control = _stratified_sample(
        control_pool, control_count, random_state=random_state + 1
    ).assign(selection_pool="control")
    selected = pd.concat([relevant, control], ignore_index=True)
    selected["_score"] = selected["record_id"].map(
        lambda value: _stable_score(str(value), random_state=random_state + 2)
    )
    selected = selected.sort_values("_score", ignore_index=True).drop(columns="_score")
    batch_core = (
        f"{batch_name}:{snapshot.dataset_sha256}:{item_count}:"
        f"{relevant_fraction:.8f}:{annotators_per_item}:{random_state}"
    )
    batch_id = "annotation:" + hashlib.sha256(batch_core.encode("utf-8")).hexdigest()

    selection_pool_counts = {
        str(key): int(value) for key, value in selected["selection_pool"].value_counts().items()
    }
    source_counts = {
        str(key): int(value) for key, value in selected["source_id"].value_counts().items()
    }
    rows: list[dict[str, object]] = []
    for article in selected.itertuples(index=False):
        item_id = "annotation-item:" + hashlib.sha256(
            f"{snapshot.dataset_sha256}:{article.record_id}".encode()
        ).hexdigest()
        for slot in range(1, annotators_per_item + 1):
            rows.append(
                {
                    "batch_id": batch_id,
                    "news_dataset_sha256": snapshot.dataset_sha256,
                    "item_id": item_id,
                    "annotation_slot": slot,
                    "record_id": article.record_id,
                    "source_id": article.source_id,
                    "published_date": article.published_date,
                    "first_available_at": pd.Timestamp(article.first_available_at).isoformat(),
                    "canonical_url": article.canonical_url,
                    "license_name": article.license_name,
                    "title": article.title,
                    "summary": article.summary,
                    "annotator_id": "",
                    "annotated_at_utc": "",
                    "relevance_label": "",
                    "direction_label": "",
                    "intensity_label": "",
                    "horizon_label": "",
                    "evidence_text": "",
                    "rationale": "",
                }
            )
    frame = pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)
    return AnnotationBatch(
        frame=frame,
        batch_id=batch_id,
        news_dataset_sha256=snapshot.dataset_sha256,
        item_count=item_count,
        rows_per_item=annotators_per_item,
        selection_pool_counts=selection_pool_counts,
        source_counts=source_counts,
    )


def _cohen_kappa(first: Iterable[object], second: Iterable[object]) -> float:
    left = pd.Series(list(first), dtype="string")
    right = pd.Series(list(second), dtype="string")
    if len(left) != len(right) or not len(left):
        raise ValueError("agreement vectors must be non-empty and have equal length")
    labels = sorted(set(left.dropna()) | set(right.dropna()))
    observed = float(np.mean(left.to_numpy() == right.to_numpy()))
    expected = sum(float((left == label).mean() * (right == label).mean()) for label in labels)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return float((observed - expected) / (1.0 - expected))


def validate_annotations(
    frame: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Validate schema, double-blind completion and inter-annotator agreement."""

    if tuple(frame.columns) != ANNOTATION_COLUMNS:
        raise ValueError("annotation columns differ from the v1 contract")
    if frame.empty:
        raise ValueError("annotation table is empty")
    if frame.duplicated(["item_id", "annotation_slot"]).any():
        raise ValueError("duplicate item_id/annotation_slot pair")
    item_sizes = frame.groupby("item_id").size()
    if item_sizes.nunique() != 1 or int(item_sizes.iloc[0]) < 2:
        raise ValueError("each item must have the same number of at least two slots")
    if not require_complete:
        return {
            "complete": False,
            "items": int(frame["item_id"].nunique()),
            "rows": len(frame),
            "rows_per_item": int(item_sizes.iloc[0]),
        }
    required = (
        "annotator_id",
        "annotated_at_utc",
        "relevance_label",
        "direction_label",
        "intensity_label",
        "horizon_label",
        "rationale",
    )
    if any(frame[column].astype("string").str.strip().eq("").any() for column in required):
        raise ValueError("completed annotations contain blank required fields")
    relevance = pd.to_numeric(frame["relevance_label"], errors="coerce")
    intensity = pd.to_numeric(frame["intensity_label"], errors="coerce")
    if relevance.isna().any() or not relevance.isin([0, 1, 2, 3]).all():
        raise ValueError("relevance_label must be one of 0, 1, 2, 3")
    if intensity.isna().any() or not intensity.isin([0, 1, 2, 3]).all():
        raise ValueError("intensity_label must be one of 0, 1, 2, 3")
    if not frame["direction_label"].isin(_DIRECTIONS).all():
        raise ValueError("direction_label has an unsupported value")
    if not frame["horizon_label"].isin(_HORIZONS).all():
        raise ValueError("horizon_label has an unsupported value")
    timestamps = pd.to_datetime(frame["annotated_at_utc"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("annotated_at_utc must contain valid UTC timestamps")
    duplicate_annotator = frame.groupby("item_id")["annotator_id"].apply(
        lambda values: values.duplicated().any()
    )
    if duplicate_annotator.any():
        raise ValueError("one annotator cannot fill multiple slots for the same item")

    ordered = frame.sort_values(["item_id", "annotation_slot"])
    first = ordered.groupby("item_id", sort=False).nth(0)
    second = ordered.groupby("item_id", sort=False).nth(1)
    return {
        "complete": True,
        "items": int(frame["item_id"].nunique()),
        "rows": len(frame),
        "rows_per_item": int(item_sizes.iloc[0]),
        "direction_cohen_kappa": _cohen_kappa(
            first["direction_label"], second["direction_label"]
        ),
        "relevance_cohen_kappa": _cohen_kappa(
            first["relevance_label"], second["relevance_label"]
        ),
        "intensity_cohen_kappa": _cohen_kappa(
            first["intensity_label"], second["intensity_label"]
        ),
        "horizon_cohen_kappa": _cohen_kappa(
            first["horizon_label"], second["horizon_label"]
        ),
    }


def merge_annotation_slots(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge independently filled slot files and reject altered source content."""

    values = list(frames)
    if len(values) < 2:
        raise ValueError("at least two independent annotation files are required")
    for frame in values:
        if tuple(frame.columns) != ANNOTATION_COLUMNS:
            raise ValueError("annotation columns differ from the v1 contract")
        slots = pd.to_numeric(frame["annotation_slot"], errors="coerce")
        if slots.isna().any() or slots.nunique() != 1:
            raise ValueError("each annotation file must contain exactly one slot")
    combined = pd.concat(values, ignore_index=True)
    if combined.duplicated(["item_id", "annotation_slot"]).any():
        raise ValueError("duplicate item_id/annotation_slot pair across files")
    static_columns = tuple(
        column
        for column in ANNOTATION_COLUMNS[: ANNOTATION_COLUMNS.index("annotator_id")]
        if column != "annotation_slot"
    )
    for column in static_columns:
        if combined.groupby("item_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"immutable annotation field changed across slots: {column}")
    validate_annotations(combined, require_complete=True)
    return combined.sort_values(["item_id", "annotation_slot"], ignore_index=True)
