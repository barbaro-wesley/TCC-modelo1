from __future__ import annotations

import json

import pytest

from vs_epl_krls.audit import _record_hash, append_audit_record, verify_audit_ledger


def test_audit_ledger_is_hash_chained_and_deterministically_verified(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    first = append_audit_record(
        ledger,
        event="forecast",
        payload={"point": 6.9},
        recorded_at_utc="2026-08-23T00:00:00+00:00",
    )
    second = append_audit_record(
        ledger,
        event="outcome",
        payload={"actual": 6.89},
        recorded_at_utc="2026-08-24T00:00:00+00:00",
    )
    records = verify_audit_ledger(ledger)
    assert [record["sequence"] for record in records] == [1, 2]
    assert second["previous_hash"] == first["record_hash"]


def test_audit_ledger_rejects_tampering_and_bad_events(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    append_audit_record(ledger, event="release", payload={"status": "candidate"})
    record = json.loads(ledger.read_text(encoding="utf-8"))
    record["payload"]["status"] = "production"
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_audit_ledger(ledger)
    with pytest.raises(ValueError, match="event"):
        append_audit_record(tmp_path / "other.jsonl", event="bad event", payload={})


def test_missing_audit_ledger_is_empty(tmp_path):
    assert verify_audit_ledger(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json\n", "invalid audit JSON"),
        ("[]\n", "invalid audit record"),
    ],
)
def test_audit_ledger_rejects_malformed_records(tmp_path, content, message):
    ledger = tmp_path / "malformed.jsonl"
    ledger.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        verify_audit_ledger(ledger)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sequence", 2, "sequence mismatch"),
        ("previous_hash", "0" * 64, "chain mismatch"),
    ],
)
def test_audit_ledger_rejects_broken_sequence_or_chain(
    tmp_path, field, value, message
):
    ledger = tmp_path / f"broken-{field}.jsonl"
    record = append_audit_record(ledger, event="release", payload={})
    record[field] = value
    unsigned = {key: value for key, value in record.items() if key != "record_hash"}
    record["record_hash"] = _record_hash(unsigned)
    ledger.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        verify_audit_ledger(ledger)


def _forecast_payload(target: str, point: float, *, sha: str = "a" * 64) -> dict:
    return {
        "artifact_sha256": sha,
        "forecast": {
            "target_date": target,
            "origin_price": 6.50,
            "point": point,
            "lower": point - 0.05,
            "upper": point + 0.05,
        },
        "decision": {"recommend_prebuy": True},
    }


def test_forecast_is_recorded_once_and_revised_when_the_artifact_changes(tmp_path):
    """Reemitir a semana com outro artefato e revisao, nao registro novo."""

    from vs_epl_krls.audit import record_forecast, verify_audit_ledger

    ledger = tmp_path / "ledger.jsonl"
    payload = _forecast_payload("2026-08-23", 6.60)

    first = record_forecast(ledger, payload, target_date="2026-08-23")
    repeated = record_forecast(ledger, payload, target_date="2026-08-23")
    revised = record_forecast(
        ledger, _forecast_payload("2026-08-23", 6.62, sha="b" * 64), target_date="2026-08-23"
    )

    assert first["event"] == "forecast"
    assert repeated is None
    assert revised["event"] == "forecast_revision"
    assert revised["payload"]["supersedes_artifact_sha256"] == "a" * 64
    assert revised["payload"]["supersedes_record_hash"] == first["record_hash"]
    events = [record["event"] for record in verify_audit_ledger(ledger)]
    assert events == ["forecast", "forecast_revision"]


def test_settlement_waits_for_the_official_value_and_counts_once(tmp_path):
    from vs_epl_krls.audit import record_forecast, settle_pending_forecast, verify_audit_ledger

    ledger = tmp_path / "ledger.jsonl"
    assert settle_pending_forecast(ledger, {}) is None  # nada emitido ainda

    record_forecast(ledger, _forecast_payload("2026-08-23", 6.60), target_date="2026-08-23")
    assert settle_pending_forecast(ledger, {"2026-08-16": 6.50}) is None  # alvo nao observado

    settled = settle_pending_forecast(
        ledger, {"2026-08-23": 6.58}, model_name="regional_rs"
    )
    payload = settled["payload"]
    assert payload["model"] == "regional_rs"
    assert payload["observed_price"] == pytest.approx(6.58)
    assert payload["absolute_error"] == pytest.approx(0.02, abs=1e-9)
    assert payload["persistence_absolute_error"] == pytest.approx(0.08, abs=1e-9)
    assert payload["interval_covered"] is True

    # Idempotente: a mesma semana nunca conta duas vezes.
    assert settle_pending_forecast(ledger, {"2026-08-23": 6.58}) is None
    assert [record["event"] for record in verify_audit_ledger(ledger)] == [
        "forecast",
        "realized",
    ]


def test_settlement_scores_the_revision_not_the_superseded_forecast(tmp_path):
    from vs_epl_krls.audit import record_forecast, settle_pending_forecast

    ledger = tmp_path / "ledger.jsonl"
    record_forecast(ledger, _forecast_payload("2026-08-23", 6.60), target_date="2026-08-23")
    record_forecast(
        ledger, _forecast_payload("2026-08-23", 6.70, sha="c" * 64), target_date="2026-08-23"
    )

    settled = settle_pending_forecast(ledger, {"2026-08-23": 6.58})

    # 6.70 e a previsao vigente; 6.60 foi superada.
    assert settled["payload"]["point"] == pytest.approx(6.70)
    assert settled["payload"]["absolute_error"] == pytest.approx(0.12, abs=1e-9)


def test_settlement_flags_an_interval_miss(tmp_path):
    from vs_epl_krls.audit import record_forecast, settle_pending_forecast

    ledger = tmp_path / "ledger.jsonl"
    record_forecast(ledger, _forecast_payload("2026-08-23", 6.60), target_date="2026-08-23")

    settled = settle_pending_forecast(ledger, {"2026-08-23": 7.40})

    assert settled["payload"]["interval_covered"] is False
