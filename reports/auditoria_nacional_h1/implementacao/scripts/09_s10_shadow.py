"""Freeze and operate the prospective S10 ARIMA-plus-VS shadow challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls import load_anp_fuel_csv
from vs_epl_krls.anp_official import extract_national_s10_observation
from vs_epl_krls.selection import S10Candidate
from vs_epl_krls.shadow import (
    S10ResidualHybridShadow,
    append_shadow_ledger,
    verify_shadow_ledger,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_candidate(path: Path) -> tuple[S10Candidate, dict[str, object]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("holdout_evaluated") is not False:
        raise RuntimeError("shadow candidate manifest must prove holdout_evaluated=false")
    if manifest.get("production_promotion_allowed") is not False:
        raise RuntimeError("research manifest must not authorize production promotion")
    validation = manifest.get("best_hybrid_validation", {})
    if validation.get("eligible_for_shadow") is not True:
        raise RuntimeError("selected hybrid did not pass the frozen shadow gate")
    candidate = S10Candidate(**manifest["best_hybrid"])
    return candidate, manifest


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze(args: argparse.Namespace) -> dict[str, object]:
    if args.artifact.exists() or args.ledger.exists() or args.freeze_manifest.exists():
        raise FileExistsError(
            "shadow freeze targets already exist; preserve them and choose new paths"
        )
    candidate, source_manifest = _load_frozen_candidate(args.candidate_manifest)
    data = load_anp_fuel_csv(args.data, products=["S10"], weekly="mean")
    history = data[["date", "price"]].copy()
    model = S10ResidualHybridShadow(candidate).fit(history)
    forecast = model.issue_forecast()
    artifact = model.save(args.artifact)
    artifact_hash = _sha256(artifact)
    candidate_manifest_hash = _sha256(args.candidate_manifest)
    freeze_payload: dict[str, object] = {
        "experiment": "prospective S10 residual hybrid shadow",
        "frozen": True,
        "retuning_allowed": False,
        "holdout_reuse_allowed": False,
        "automatic_promotion_allowed": False,
        "minimum_future_outcomes": 26,
        "preferred_future_outcomes": 52,
        "candidate": candidate.__dict__,
        "candidate_fingerprint": model.candidate_fingerprint_,
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": candidate_manifest_hash,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_hash,
        "freeze_cutoff": model.freeze_cutoff_,
        "data_fingerprint": model.data_fingerprint_,
        "development_end_index": source_manifest["development_end_index"],
    }
    freeze_record = append_shadow_ledger(
        args.ledger, event="freeze", payload=freeze_payload
    )
    forecast_record = append_shadow_ledger(
        args.ledger,
        event="forecast",
        payload={
            **forecast.as_dict(),
            "artifact_sha256": artifact_hash,
            "candidate_manifest_sha256": candidate_manifest_hash,
        },
    )
    ledger_records = verify_shadow_ledger(args.ledger)
    payload: dict[str, object] = {
        **freeze_payload,
        "forecast": forecast.as_dict(),
        "health": model.health().as_dict(),
        "promotion_report": model.promotion_report(),
        "ledger": str(args.ledger.resolve()),
        "ledger_records": len(ledger_records),
        "ledger_head_hash": ledger_records[-1]["record_hash"],
        "freeze_record_hash": freeze_record["record_hash"],
        "forecast_record_hash": forecast_record["record_hash"],
    }
    _write_json(args.freeze_manifest, payload)
    _write_json(args.output, payload)
    return payload


def operate(args: argparse.Namespace) -> dict[str, object]:
    if not args.freeze_manifest.exists():
        raise FileNotFoundError("freeze manifest is required to operate the shadow")
    frozen = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    fingerprint = str(frozen["candidate_fingerprint"])
    ledger = verify_shadow_ledger(args.ledger)
    if len(ledger) < 2 or ledger[-1]["event"] != "forecast":
        raise RuntimeError("shadow ledger must end with a pending forecast")
    ledger_artifact_hash = ledger[-1]["payload"].get("artifact_sha256")
    if not ledger_artifact_hash:
        raise RuntimeError("shadow ledger does not identify the pending artifact hash")
    if (
        args.expected_sha256 is not None
        and args.expected_sha256.lower() != str(ledger_artifact_hash).lower()
    ):
        raise RuntimeError("provided SHA-256 does not match the shadow ledger")
    model = S10ResidualHybridShadow.load(
        args.artifact,
        expected_sha256=str(ledger_artifact_hash),
        expected_candidate_fingerprint=fingerprint,
    )
    pending_id = ledger[-1]["payload"]["forecast_id"]
    if model.pending_forecast_ is None or model.pending_forecast_.forecast_id != pending_id:
        raise RuntimeError("artifact pending forecast does not match ledger head")

    has_date = args.update_date is not None
    has_price = args.update_price is not None
    if has_date != has_price:
        raise ValueError("--update-date and --update-price must be supplied together")
    updated = has_date and has_price
    evaluation = None
    source_observation = None
    source_workbook = getattr(args, "source_workbook", None)
    if updated and source_workbook is not None:
        source_observation = extract_national_s10_observation(
            source_workbook,
            source_url=getattr(args, "source_url", None),
        )
        if source_observation.week_start != str(args.update_date):
            raise ValueError("official ANP week does not match --update-date")
        if abs(source_observation.price_brl_per_liter - float(args.update_price)) > 1e-12:
            raise ValueError("official ANP S10 price does not match --update-price")
    artifact_path = args.artifact
    artifact_hash = _sha256(args.artifact)
    if updated:
        if args.output_artifact is None:
            raise ValueError("--output-artifact is required for a shadow update")
        if args.output_artifact.resolve() == args.artifact.resolve():
            raise ValueError("shadow updates must preserve the previous artifact")
        evaluation = model.update_one(args.update_date, args.update_price)
        next_forecast = model.issue_forecast()
        artifact_path = model.save(args.output_artifact)
        artifact_hash = _sha256(artifact_path)
        append_shadow_ledger(
            args.ledger,
            event="outcome",
            payload={
                **evaluation,
                "source_artifact": str(args.artifact.resolve()),
                "source_observation": (
                    source_observation.as_dict() if source_observation else None
                ),
            },
        )
        append_shadow_ledger(
            args.ledger,
            event="forecast",
            payload={
                **next_forecast.as_dict(),
                "artifact": str(artifact_path.resolve()),
                "artifact_sha256": artifact_hash,
            },
        )
    else:
        next_forecast = model.pending_forecast_

    ledger = verify_shadow_ledger(args.ledger)
    payload: dict[str, object] = {
        "updated": bool(updated),
        "evaluation": evaluation,
        "forecast": next_forecast.as_dict(),
        "health": model.health().as_dict(),
        "promotion_report": model.promotion_report(),
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_hash,
        "source_observation": (
            source_observation.as_dict() if source_observation else None
        ),
        "candidate_fingerprint": model.candidate_fingerprint_,
        "ledger_records": len(ledger),
        "ledger_head_hash": ledger[-1]["record_hash"],
    }
    _write_json(args.output, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, object]:
    return freeze(args) if args.freeze else operate(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "anp_semanal_desde_2013.xlsx",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_next"
        / "next_challenger_manifest.json",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "s10_shadow_hybrid_v1.joblib",
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument("--update-date")
    parser.add_argument("--update-price", type=float)
    parser.add_argument(
        "--source-workbook",
        type=Path,
        help="planilha semanal oficial da ANP para validar data, preço e proveniência",
    )
    parser.add_argument("--source-url", help="URL institucional da planilha oficial")
    parser.add_argument("--output-artifact", type=Path)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_shadow"
        / "shadow_ledger.jsonl",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_shadow"
        / "freeze_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_shadow"
        / "latest_status.json",
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
