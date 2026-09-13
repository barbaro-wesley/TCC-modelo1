from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.news_pressure import (
    NEWS_PRESSURE_FEATURES,
    NewsPressureConfig,
    OnlineNewsPressureClassifier,
    augment_supervised_with_pressure,
    build_weekly_news_documents,
    generate_prequential_pressure_features,
    load_news_corpus,
    market_pressure_labels,
)
from vs_epl_krls.selection import S10Supervised


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _record(
    *,
    record_id: str,
    first_available_at: str,
    title: str,
    relevance: float,
) -> dict[str, object]:
    content_hash = hashlib.sha256(title.encode()).hexdigest()
    return {
        "availability_basis": "published_date_end_of_day",
        "canonical_url": f"https://example.test/{record_id}",
        "content_sha256": content_hash,
        "first_available_at": first_available_at,
        "impact": {
            "categories": ["supply"],
            "confidence": 0.5,
            "direction": "up",
            "evidence_terms": ["diesel"],
            "horizon": "short",
            "intensity": 0.4,
            "model_version": "fixture.v1",
            "relevance_s10": relevance,
        },
        "language": "pt-BR",
        "license_name": "fixture",
        "provenance_ids": ["http:fixture"],
        "published_at": None,
        "published_date": first_available_at[:10],
        "record_id": f"news2:{record_id}",
        "response_sha256": "4" * 64,
        "retrieved_at": "2024-02-01T00:00:00Z",
        "schema_version": "news-record.v2",
        "source_id": "anp_news",
        "storage_scope": "metadata_summary",
        "summary": "Resumo sobre abastecimento e preço.",
        "title": title,
        "updated_at": None,
    }


def _corpus(tmp_path: Path, *, tamper: bool = False) -> Path:
    records = [
        _record(
            record_id="a",
            first_available_at="2024-01-03T03:00:00Z",
            title="Oferta de diesel cresce",
            relevance=0.8,
        ),
        _record(
            record_id="b",
            first_available_at="2024-01-07T03:00:00Z",
            title="Notícia publicada depois do corte",
            relevance=0.0,
        ),
    ]
    payload = b"".join(_canonical(record) + b"\n" for record in records)
    core = {
        "config_sha256": "1" * 64,
        "dataset_type": "news",
        "files": {"records.jsonl": hashlib.sha256(payload).hexdigest()},
        "manifest_version": "dataset-manifest.v2",
        "parent_provenance_dataset_sha256": "2" * 64,
        "pipeline_version": "temporal-impact-collection.v1",
        "record_count": len(records),
        "retrieved_through": "2024-02-01T00:00:00Z",
        "schema_version": "news-record.v2",
        "source_catalog_sha256": "3" * 64,
        "unavailable_count": 0,
    }
    digest = hashlib.sha256(_canonical(core)).hexdigest()
    version = tmp_path / "news-record" / "v2"
    snapshot = version / "snapshots" / digest
    snapshot.mkdir(parents=True)
    (snapshot / "records.jsonl").write_bytes(payload + (b"x" if tamper else b""))
    (snapshot / "manifest.json").write_text(
        json.dumps({**core, "dataset_sha256": digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pointer_core = {
        "dataset_sha256": digest,
        "pointer_version": "dataset-pointer.v1",
        "relative_path": f"snapshots/{digest}",
        "schema_version": "news-record.v2",
    }
    (version / "latest.json").write_text(
        json.dumps(
            {
                **pointer_core,
                "pointer_sha256": hashlib.sha256(_canonical(pointer_core)).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return version


def _supervised(*, future_shift: float = 0.0) -> S10Supervised:
    dates = pd.date_range("2024-01-07", periods=8, freq="7D").to_numpy()
    origin = np.array([4.00, 4.02, 4.01, 4.01, 4.03, 4.02, 4.05, 4.04])
    target = np.array([4.02, 4.01, 4.01, 4.03, 4.02, 4.05, 4.04, 4.06])
    target[4:] += future_shift
    return S10Supervised(
        x=origin[:, None],
        target_price=target,
        origin_price=origin,
        dates=dates,
        target_dates=dates + np.timedelta64(7, "D"),
        feature_names=("price",),
        horizon=1,
    )


def test_load_news_corpus_verifies_snapshot_and_rejects_tampering(tmp_path: Path) -> None:
    snapshot = load_news_corpus(_corpus(tmp_path / "ok"))
    assert len(snapshot.articles) == 2
    assert snapshot.dataset_sha256
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_news_corpus(_corpus(tmp_path / "tampered", tamper=True))


def test_weekly_documents_respect_availability_cutoff_and_relevance(tmp_path: Path) -> None:
    snapshot = load_news_corpus(_corpus(tmp_path))
    documents = build_weekly_news_documents(
        snapshot,
        ["2024-01-07", "2024-01-14"],
        lookback_days=14,
        min_relevance_s10=0.1,
    )
    first = documents.frame.iloc[0]
    assert first["article_count"] == 1
    assert "Oferta de diesel" in first["document"]
    assert "depois do corte" not in first["document"]
    assert documents.frame.iloc[1]["article_count"] == 1


def test_market_pressure_labels_apply_neutral_band() -> None:
    labels = market_pressure_labels(_supervised(), neutral_threshold=0.005)
    assert labels.tolist() == [1, -1, 0, 1, -1, 1, -1, 1]


def test_online_classifier_is_deterministic_picklable_and_normalized() -> None:
    config = NewsPressureConfig(min_training_samples=3, n_features=256)
    classifier = OnlineNewsPressureClassifier(config)
    for document, label in [("oferta cai", 1), ("oferta sobe", -1), ("estável", 0)]:
        classifier.learn_one(document, label)
    expected = classifier.predict_proba_one("oferta de diesel cai")
    restored = pickle.loads(pickle.dumps(classifier))
    observed = restored.predict_proba_one("oferta de diesel cai")
    assert np.allclose(expected, observed)
    assert observed.sum() == pytest.approx(1.0)
    assert np.all((observed >= 0.0) & (observed <= 1.0))


def test_prequential_features_do_not_change_when_only_future_labels_change(
    tmp_path: Path,
) -> None:
    snapshot = load_news_corpus(_corpus(tmp_path))
    base = _supervised()
    changed_future = _supervised(future_shift=1.0)
    documents = build_weekly_news_documents(snapshot, base.dates, lookback_days=28)
    config = NewsPressureConfig(min_training_samples=3, n_features=256)
    first = generate_prequential_pressure_features(base, documents, config=config)
    second = generate_prequential_pressure_features(changed_future, documents, config=config)
    columns = list(NEWS_PRESSURE_FEATURES)
    assert np.allclose(
        first.frame.loc[:4, columns].to_numpy(),
        second.frame.loc[:4, columns].to_numpy(),
    )
    assert first.frame["pressure_training_samples"].tolist() == list(range(8))
    assert np.allclose(
        first.frame[
            [
                "news_pressure_down_probability",
                "news_pressure_neutral_probability",
                "news_pressure_up_probability",
            ]
        ].sum(axis=1),
        1.0,
    )


def test_augment_supervised_with_pressure_preserves_targets(tmp_path: Path) -> None:
    snapshot = load_news_corpus(_corpus(tmp_path))
    base = _supervised()
    documents = build_weekly_news_documents(snapshot, base.dates)
    pressure = generate_prequential_pressure_features(
        base,
        documents,
        config=NewsPressureConfig(min_training_samples=3, n_features=256),
    )
    augmented = augment_supervised_with_pressure(base, pressure)
    assert augmented.x.shape == (8, 1 + len(NEWS_PRESSURE_FEATURES))
    assert np.array_equal(augmented.target_price, base.target_price)
    assert np.array_equal(augmented.target_dates, base.target_dates)
