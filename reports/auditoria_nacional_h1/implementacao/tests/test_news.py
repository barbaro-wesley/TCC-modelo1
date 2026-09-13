from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from vs_epl_krls.news import (
    NEWS_CORE_FEATURES,
    augment_supervised_with_news,
    load_weekly_news_features,
)
from vs_epl_krls.selection import S10Supervised


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _snapshot(tmp_path: Path, *, tamper: bool = False) -> Path:
    version = tmp_path / "weekly-signal" / "v3"
    signal = {
        "active_source_count": 4,
        "article_count": 2,
        "category_counts": {
            "biodiesel": 0,
            "demand": 0,
            "logistics_strike": 0,
            "oil_brent": 1,
            "refining_petrobras": 0,
            "regulation": 1,
            "supply": 1,
            "tax": 0,
            "usd_brl": 0,
        },
        "direction_counts": {"down": 0, "neutral": 0, "uncertain": 1, "up": 1},
        "exact_duplicate_suppressed_count": 1,
        "forecast_at": "2024-01-07T00:00:00Z",
        "lookback_days": 28,
        "mean_confidence": 0.5,
        "mean_intensity": 0.4,
        "mean_relevance_s10": 0.6,
        "missing_summary_count": 1,
        "no_news": False,
        "observed_source_count": 2,
        "revision_superseded_count": 0,
        "schema_version": "weekly-signal.v3",
        "signal_id": "weekly3:" + "a" * 64,
        "source_coverage": 0.5,
        "target_week_start": "2024-01-14",
        "unavailable_count": 0,
        "week_convention": "anp_sunday",
    }
    payload = (_canonical(signal) + b"\n")
    core = {
        "config_sha256": "1" * 64,
        "dataset_type": "weekly",
        "files": {"signals.jsonl": hashlib.sha256(payload).hexdigest()},
        "generated_through_forecast_at": "2024-01-07T00:00:00Z",
        "manifest_version": "dataset-manifest.v3",
        "parent_news_dataset_sha256": "2" * 64,
        "pipeline_version": "temporal-impact-aggregation.v2",
        "record_count": 1,
        "schema_version": "weekly-signal.v3",
        "source_catalog_sha256": "3" * 64,
    }
    digest = hashlib.sha256(_canonical(core)).hexdigest()
    snapshot = version / "snapshots" / digest
    snapshot.mkdir(parents=True)
    (snapshot / "signals.jsonl").write_bytes(payload + (b"x" if tamper else b""))
    (snapshot / "manifest.json").write_text(
        json.dumps({**core, "dataset_sha256": digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pointer_core = {
        "dataset_sha256": digest,
        "pointer_version": "dataset-pointer.v1",
        "relative_path": f"snapshots/{digest}",
        "schema_version": "weekly-signal.v3",
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


def test_load_weekly_news_features_verifies_and_derives_features(tmp_path: Path) -> None:
    result = load_weekly_news_features(_snapshot(tmp_path), horizon_weeks=1)
    assert result.dataset_sha256
    assert result.frame.loc[0, "news_direction_balance"] == pytest.approx(0.5)
    assert result.frame.loc[0, "news_category_supply_share"] == pytest.approx(0.5)
    assert result.frame.loc[0, "news_source_coverage"] == pytest.approx(0.5)


def test_load_weekly_news_features_rejects_tampering(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_weekly_news_features(_snapshot(tmp_path, tamper=True), horizon_weeks=1)


def test_augment_supervised_preserves_targets_and_aligns_origins(tmp_path: Path) -> None:
    snapshot = load_weekly_news_features(_snapshot(tmp_path), horizon_weeks=1)
    base = S10Supervised(
        x=np.array([[1.0, 2.0]]),
        target_price=np.array([5.0]),
        origin_price=np.array([4.0]),
        dates=np.array(["2024-01-07"], dtype="datetime64[ns]"),
        target_dates=np.array(["2024-01-14"], dtype="datetime64[ns]"),
        feature_names=("a", "b"),
        horizon=1,
    )
    augmented = augment_supervised_with_news(base, snapshot)
    assert augmented.x.shape == (1, 2 + len(NEWS_CORE_FEATURES))
    assert np.array_equal(augmented.target_price, base.target_price)
    assert np.array_equal(augmented.target_dates, base.target_dates)


def test_augment_supervised_rejects_missing_origin(tmp_path: Path) -> None:
    snapshot = load_weekly_news_features(_snapshot(tmp_path), horizon_weeks=1)
    base = S10Supervised(
        x=np.array([[1.0]]),
        target_price=np.array([5.0]),
        origin_price=np.array([4.0]),
        dates=np.array(["2024-02-04"], dtype="datetime64[ns]"),
        target_dates=np.array(["2024-02-11"], dtype="datetime64[ns]"),
        feature_names=("a",),
        horizon=1,
    )
    with pytest.raises(ValueError, match="missing 1 supervised origins"):
        augment_supervised_with_news(base, snapshot)
