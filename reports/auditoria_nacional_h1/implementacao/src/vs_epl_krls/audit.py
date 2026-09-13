"""Tamper-evident append-only ledgers for production evidence.

The ledger is deliberately small and dependency-free.  It does not replace a
managed audit-log service, but it makes local releases and forecasts
independently verifiable before they are exported to one.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record_without_hash).encode("utf-8")).hexdigest()


def verify_audit_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Return verified records or raise when any link/content was modified."""

    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid audit JSON at line {line_number}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"invalid audit record at line {line_number}")
            observed_hash = record.get("record_hash")
            unsigned = {key: value for key, value in record.items() if key != "record_hash"}
            if observed_hash != _record_hash(unsigned):
                raise RuntimeError(f"audit record hash mismatch at line {line_number}")
            if unsigned.get("sequence") != len(records) + 1:
                raise RuntimeError(f"audit sequence mismatch at line {line_number}")
            if unsigned.get("previous_hash") != previous_hash:
                raise RuntimeError(f"audit chain mismatch at line {line_number}")
            records.append(record)
            previous_hash = str(observed_hash)
    return records


def append_audit_record(
    path: str | Path,
    *,
    event: str,
    payload: dict[str, Any],
    recorded_at_utc: str | None = None,
) -> dict[str, Any]:
    """Append one canonical, hash-chained event after verifying existing data."""

    if not event or any(character.isspace() for character in event):
        raise ValueError("event must be a non-empty token without whitespace")
    destination = Path(path)
    records = verify_audit_ledger(destination)
    unsigned: dict[str, Any] = {
        "sequence": len(records) + 1,
        "recorded_at_utc": recorded_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
        "previous_hash": records[-1]["record_hash"] if records else None,
    }
    record = {**unsigned, "record_hash": _record_hash(unsigned)}
    encoded = _canonical_json(record) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
    verify_audit_ledger(destination)
    return record



#: Eventos que registram uma previsao emitida e ainda pendente de realizacao.
FORECAST_EVENTS = ("forecast", "forecast_revision")


def settle_pending_forecast(
    path: str | Path,
    observed: Mapping[str, float],
    *,
    model_name: str = "model",
) -> dict[str, Any] | None:
    """Pontua a previsao pendente assim que a semana-alvo dela e observada.

    Sem isto a contagem prospectiva nao existe: o JSON de saida e sobrescrito a
    cada execucao, entao a previsao da semana passada desaparece antes de poder
    ser comparada com o realizado.

    ``observed`` mapeia data ISO para o preco oficial daquela semana.  A funcao e
    idempotente: uma semana ja liquidada nunca e contada de novo, e nada e
    gravado enquanto o valor oficial nao chega.  Devolve o registro criado, ou
    ``None`` quando nao havia o que liquidar.
    """

    records = verify_audit_ledger(path)
    issued = [record for record in records if record["event"] in FORECAST_EVENTS]
    if not issued:
        return None
    pending = issued[-1]
    forecast = pending["payload"]["forecast"]
    target = str(forecast["target_date"])
    if any(
        record["event"] == "realized" and str(record["payload"]["target_date"]) == target
        for record in records
    ):
        return None
    if target not in observed:
        return None

    actual = float(observed[target])
    point = float(forecast["point"])
    origin_price = float(forecast["origin_price"])
    decision = pending["payload"].get("decision") or {}
    return append_audit_record(
        path,
        event="realized",
        payload={
            "model": model_name,
            "target_date": target,
            "observed_price": actual,
            "point": point,
            "absolute_error": round(abs(actual - point), 6),
            "persistence_absolute_error": round(abs(actual - origin_price), 6),
            "interval_covered": bool(
                float(forecast["lower"]) <= actual <= float(forecast["upper"])
            ),
            "recommended_prebuy": bool(decision.get("recommend_prebuy", False)),
            "forecast_record_hash": pending["record_hash"],
        },
    )


def record_forecast(
    path: str | Path, payload: dict[str, Any], *, target_date: str
) -> dict[str, Any] | None:
    """Registra a previsao, ou uma revisao quando o artefato daquela semana mudou.

    Reemitir a mesma semana com outro artefato e uma revisao, nao um registro
    novo: encadear como revisao preserva a contagem prospectiva e deixa a troca
    visivel em vez de sobrescrever a evidencia anterior.  Reexecutar com saida
    identica nao acrescenta nada.
    """

    issued = [
        record
        for record in verify_audit_ledger(path)
        if record["event"] in FORECAST_EVENTS
        and str(record["payload"]["forecast"]["target_date"]) == str(target_date)
    ]
    if not issued:
        return append_audit_record(path, event="forecast", payload=payload)
    previous = issued[-1]["payload"].get("artifact_sha256")
    if previous == payload.get("artifact_sha256"):
        return None
    return append_audit_record(
        path,
        event="forecast_revision",
        payload={
            **payload,
            "supersedes_artifact_sha256": previous,
            "supersedes_record_hash": issued[-1]["record_hash"],
        },
    )
