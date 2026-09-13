from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.selection import S10Candidate
from vs_epl_krls.shadow import (
    S10ResidualHybridShadow,
    append_shadow_ledger,
    verify_shadow_ledger,
)


class DummyARIMA:
    """Small pickle-safe causal stand-in for unit tests."""

    def __init__(self, last: float):
        self.last = float(last)
        self.aic = 0.0

    def append(self, values, refit=False):
        return DummyARIMA(float(np.asarray(values).ravel()[-1]))

    def forecast(self, steps=1):
        return np.repeat(self.last, steps)


def _history(n: int = 120) -> pd.DataFrame:
    index = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=n, freq="7D"),
            "price": 5.2 + 0.004 * index + 0.04 * np.sin(index / 8.0),
        }
    )


def _candidate(identifier: str = "shadow") -> S10Candidate:
    return S10Candidate(
        candidate_id=identifier,
        feature_set="dynamics",
        target_mode="delta",
        alpha=0.26,
        beta_initial=0.18,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=0.5,
        kernel_sigma=0.15,
        regularization=1e-3,
        max_dictionary_size=5,
        max_rules=5,
        beta_min=0.001,
        beta_recovery_rate=0.1,
        dictionary_usage_decay=0.995,
        residual_correction_weight=0.5,
        residual_correction_limit=0.1,
        center_update="paper",
        threshold_policy="dynamic",
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )


@pytest.fixture
def shadow(monkeypatch) -> S10ResidualHybridShadow:
    monkeypatch.setattr(
        S10ResidualHybridShadow,
        "_fit_arima",
        staticmethod(lambda values: DummyARIMA(float(values[-1]))),
    )
    return S10ResidualHybridShadow(_candidate()).fit(_history())


def test_fit_uses_causal_residuals_and_issues_bounded_idempotent_forecast(shadow):
    # Raw origins begin after 12 warm-up rows. Causal ARIMA begins at raw row 51
    # and the last labelled origin is raw row 118: 68 residuals.
    assert shadow.baseline_training_samples_ == 68
    assert shadow.metadata()["freeze_cutoff"] == str(_history()["date"].iloc[-1].date())
    prediction = shadow.predict_next()
    assert shadow.pending_forecast_ is None
    issued = shadow.issue_forecast()
    assert shadow.issue_forecast() == issued
    assert prediction == issued
    assert abs(issued.applied_correction) <= 0.1
    assert issued.point == pytest.approx(issued.base_point + issued.applied_correction)
    assert len(issued.forecast_id) == len(issued.candidate_fingerprint) == 64
    assert shadow.health().pending_target_date == issued.target_date


def test_update_scores_exact_pending_forecast_before_learning(shadow):
    issued = shadow.issue_forecast()
    with pytest.raises(ValueError, match="pending target_date"):
        shadow.update_one("2030-01-01", 5.8)
    actual = issued.point + 0.02
    result = shadow.update_one(issued.target_date, actual)
    assert result["hybrid_error"] == pytest.approx(0.02)
    assert result["arima_error"] == pytest.approx(actual - issued.base_point)
    assert shadow.health().n_outcomes == 1
    assert shadow.promotion_report()["status"] == "collecting"
    assert shadow.promotion_report()["weeks_remaining"] == 25
    with pytest.raises(RuntimeError, match="issue_forecast"):
        shadow.update_one(issued.target_date, actual)
    following = shadow.issue_forecast()
    assert pd.Timestamp(following.target_date) - pd.Timestamp(issued.target_date) == pd.Timedelta(days=7)


def test_roundtrip_hash_and_candidate_fingerprint_are_enforced(shadow, tmp_path):
    shadow.issue_forecast()
    path = shadow.save(tmp_path / "shadow.joblib")
    digest = S10ResidualHybridShadow._file_sha256(path)
    restored = S10ResidualHybridShadow.load(
        path,
        expected_sha256=digest,
        expected_candidate_fingerprint=shadow.candidate_fingerprint_,
    )
    assert restored.pending_forecast_ == shadow.pending_forecast_
    with pytest.raises(RuntimeError, match="SHA-256"):
        S10ResidualHybridShadow.load(path, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="candidate fingerprint"):
        S10ResidualHybridShadow.load(path, expected_candidate_fingerprint="0" * 64)
    shadow.artifact_version_ = "0.9.0"
    old = tmp_path / "old.joblib"
    joblib.dump(shadow, old)
    with pytest.raises(RuntimeError, match="unsupported"):
        S10ResidualHybridShadow.load(old)


def test_hash_chained_ledger_detects_tampering(tmp_path):
    ledger = tmp_path / "shadow.jsonl"
    first = append_shadow_ledger(ledger, event="freeze", payload={"candidate": "abc"})
    second = append_shadow_ledger(ledger, event="forecast", payload={"point": 6.1})
    verified = verify_shadow_ledger(ledger)
    assert [row["sequence"] for row in verified] == [1, 2]
    assert second["previous_hash"] == first["record_hash"]
    lines = ledger.read_text(encoding="utf-8").splitlines()
    changed = json.loads(lines[0])
    changed["payload"]["candidate"] = "tampered"
    lines[0] = json.dumps(changed)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="record hash"):
        verify_shadow_ledger(ledger)


def test_promotion_report_never_auto_promotes_and_requires_frozen_gates(shadow):
    shadow.evaluations_ = []
    for index in range(26):
        actual = 6.0 + index * 0.01
        shadow.evaluations_.append(
            {
                "actual": actual,
                "hybrid": actual,
                "arima": actual + (0.08 if index % 2 else -0.08),
                "persistence": actual + (0.12 if index % 2 else -0.12),
            }
        )
    report = shadow.promotion_report()
    assert report["status"] == "eligible_for_human_review"
    assert report["automatic_promotion_allowed"] is False
    assert report["gates"]["minimum_evidence"] is True
    assert report["gates"]["preferred_evidence"] is False
    assert report["rmse_ratio_vs_arima"] == 0.0
    with pytest.raises(ValueError, match="thresholds"):
        shadow.promotion_report(minimum_outcomes=12)


@pytest.mark.parametrize(
    "candidate, kwargs, message",
    [
        (_candidate(), {"min_arima_history": 19}, "timing"),
        (
            S10Candidate(**{**_candidate().__dict__, "target_mode": "level"}),
            {},
            "target_mode",
        ),
        (
            S10Candidate(**{**_candidate().__dict__, "feature_set": "exogenous"}),
            {},
            "does not ingest",
        ),
    ],
)
def test_shadow_constructor_rejects_invalid_contract(candidate, kwargs, message):
    with pytest.raises(ValueError, match=message):
        S10ResidualHybridShadow(candidate, **kwargs)
