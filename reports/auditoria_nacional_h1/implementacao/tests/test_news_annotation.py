from __future__ import annotations

import pandas as pd
import pytest

from vs_epl_krls.news_annotation import (
    build_annotation_batch,
    merge_annotation_slots,
    validate_annotations,
)
from vs_epl_krls.news_pressure import NewsCorpusSnapshot


def _snapshot() -> NewsCorpusSnapshot:
    rows = []
    for index in range(30):
        rows.append(
            {
                "record_id": f"news2:{index:02d}",
                "canonical_url": f"https://example.test/{index}",
                "first_available_at": pd.Timestamp("2020-01-01", tz="UTC")
                + pd.Timedelta(days=30 * index),
                "published_date": str(
                    (pd.Timestamp("2020-01-01") + pd.Timedelta(days=30 * index)).date()
                ),
                "title": f"Notícia {index}",
                "summary": "Resumo",
                "source_id": "anp_news" if index % 2 else "mme_news",
                "license_name": "fixture",
                "content_sha256": f"{index:064x}",
                "relevance_s10": 0.5 if index < 20 else 0.0,
                "machine_direction": "up",
                "machine_confidence": 0.5,
                "machine_intensity": 0.4,
                "machine_categories": ("supply",),
            }
        )
    return NewsCorpusSnapshot(
        articles=pd.DataFrame(rows),
        dataset_sha256="1" * 64,
        parent_provenance_dataset_sha256="2" * 64,
        source_catalog_sha256="3" * 64,
        retrieved_through=pd.Timestamp("2024-01-01", tz="UTC"),
    )


def test_annotation_batch_is_deterministic_stratified_and_double_blind() -> None:
    first = build_annotation_batch(
        _snapshot(), item_count=12, relevant_fraction=2 / 3, random_state=7
    )
    second = build_annotation_batch(
        _snapshot(), item_count=12, relevant_fraction=2 / 3, random_state=7
    )
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert first.item_count == 12
    assert len(first.frame) == 24
    assert first.frame["item_id"].nunique() == 12
    assert first.frame.groupby("item_id")["annotation_slot"].nunique().eq(2).all()
    assert first.selection_pool_counts["lexicon_relevant"] == 8
    assert "machine_direction" not in first.frame
    assert validate_annotations(first.frame, require_complete=False)["items"] == 12


def test_completed_annotations_report_perfect_agreement() -> None:
    batch = build_annotation_batch(_snapshot(), item_count=10, random_state=8)
    frame = batch.frame.copy()
    frame["annotator_id"] = frame["annotation_slot"].map({1: "ana", 2: "bruno"})
    frame["annotated_at_utc"] = "2026-08-23T12:00:00Z"
    frame["relevance_label"] = "2"
    frame["direction_label"] = "up"
    frame["intensity_label"] = "1"
    frame["horizon_label"] = "2w"
    frame["rationale"] = "A notícia altera oferta ou custo do diesel."
    result = validate_annotations(frame)
    assert result["direction_cohen_kappa"] == pytest.approx(1.0)
    assert result["relevance_cohen_kappa"] == pytest.approx(1.0)


def test_annotation_validation_rejects_invalid_or_same_annotator() -> None:
    batch = build_annotation_batch(_snapshot(), item_count=10, random_state=9)
    frame = batch.frame.copy()
    frame["annotator_id"] = "same"
    frame["annotated_at_utc"] = "2026-08-23T12:00:00Z"
    frame["relevance_label"] = "9"
    frame["direction_label"] = "up"
    frame["intensity_label"] = "1"
    frame["horizon_label"] = "2w"
    frame["rationale"] = "Racional"
    with pytest.raises(ValueError, match="relevance_label"):
        validate_annotations(frame)
    frame["relevance_label"] = "2"
    with pytest.raises(ValueError, match="one annotator"):
        validate_annotations(frame)


def test_merge_annotation_slots_rejects_source_edits() -> None:
    batch = build_annotation_batch(_snapshot(), item_count=10, random_state=10)
    completed = batch.frame.copy()
    completed["annotator_id"] = completed["annotation_slot"].map({1: "ana", 2: "bruno"})
    completed["annotated_at_utc"] = "2026-08-23T12:00:00Z"
    completed["relevance_label"] = "2"
    completed["direction_label"] = "up"
    completed["intensity_label"] = "1"
    completed["horizon_label"] = "2w"
    completed["rationale"] = "Racional"
    first = completed[completed["annotation_slot"] == 1].copy()
    second = completed[completed["annotation_slot"] == 2].copy()
    merged = merge_annotation_slots([first, second])
    assert len(merged) == 20
    second.loc[second.index[0], "title"] = "Texto alterado"
    with pytest.raises(ValueError, match="immutable annotation field"):
        merge_annotation_slots([first, second])
