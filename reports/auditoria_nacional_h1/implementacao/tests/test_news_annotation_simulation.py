from __future__ import annotations

import pandas as pd
import pytest

from vs_epl_krls.news_annotation import ANNOTATION_COLUMNS, merge_annotation_slots
from vs_epl_krls.news_annotation_simulation import (
    simulate_annotation_slot,
    simulate_label,
)


def _blank(slot: int) -> pd.DataFrame:
    texts = (
        (
            "ANP anuncia redução de imposto sobre diesel S10",
            "A medida reduz o custo do combustível.",
        ),
        ("Museu abre nova exposição", "Evento cultural começa nesta semana."),
        (
            "Refinaria reduz produção após greve",
            "A paralisação reduz oferta de óleo diesel.",
        ),
    )
    rows = []
    for index, (title, summary) in enumerate(texts):
        row = {column: "" for column in ANNOTATION_COLUMNS}
        row.update(
            {
                "batch_id": "annotation:test",
                "news_dataset_sha256": "1" * 64,
                "item_id": f"item:{index}",
                "annotation_slot": str(slot),
                "record_id": f"news2:{index}",
                "source_id": "anp_news",
                "published_date": "2026-01-01",
                "first_available_at": "2026-01-02T03:00:00Z",
                "canonical_url": f"https://example.test/{index}",
                "license_name": "fixture",
                "title": title,
                "summary": summary,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS)


def test_simulated_personas_are_causal_deterministic_and_mergeable() -> None:
    first = simulate_annotation_slot(_blank(1), persona="supply_cost")
    second = simulate_annotation_slot(_blank(2), persona="procurement_risk")
    repeated = simulate_annotation_slot(_blank(1), persona="supply_cost")
    pd.testing.assert_frame_equal(first, repeated)
    assert first.loc[0, "direction_label"] == "down"
    assert first.loc[1, "relevance_label"] == "0"
    assert first.loc[2, "direction_label"] == "up"
    assert first["annotator_id"].iat[0] != second["annotator_id"].iat[0]
    merged = merge_annotation_slots([first, second])
    assert len(merged) == 6


def test_simulation_rejects_prefilled_queue() -> None:
    frame = _blank(1)
    frame.loc[0, "direction_label"] = "up"
    with pytest.raises(ValueError, match="untouched blank"):
        simulate_annotation_slot(frame, persona="supply_cost")


def test_persona_validation_and_uncertain_sector_signal() -> None:
    label = simulate_label(
        "Petrobras publica estudo sobre petróleo e câmbio",
        persona="procurement_risk",
    )
    assert label.relevance >= 1
    assert label.direction in {"neutral", "uncertain"}
    calibrated = simulate_label(
        "Petrobras publica estudo sobre petróleo e câmbio",
        persona="procurement_risk_calibrated",
    )
    assert calibrated.relevance >= label.relevance
    with pytest.raises(ValueError, match="unsupported simulation persona"):
        simulate_label("diesel", persona="invalid")  # type: ignore[arg-type]
