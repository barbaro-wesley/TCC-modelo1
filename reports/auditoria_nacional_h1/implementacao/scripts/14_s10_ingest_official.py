"""Create an immutable S10 release from one official ANP weekly workbook."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.anp_official import extract_national_s10_observation, sha256_file
from vs_epl_krls.audit import append_audit_record, verify_audit_ledger
from vs_epl_krls.production import S10ProductionForecaster


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_artifact.exists() or args.release_manifest.exists():
        raise FileExistsError("immutable release targets already exist")
    prior_manifest = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    expected_hash = prior_manifest.get("artifact_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("prior manifest does not contain a valid artifact_sha256")
    model = S10ProductionForecaster.load(
        args.artifact,
        expected_sha256=expected_hash,
    )
    issued = model.predict_next()
    observation = extract_national_s10_observation(
        args.source_workbook,
        source_url=args.source_url,
    )
    if observation.week_start != issued.target_date:
        raise ValueError(
            f"official week {observation.week_start} does not match pending target "
            f"{issued.target_date}"
        )

    existing = verify_audit_ledger(args.ledger)
    if not existing:
        append_audit_record(
            args.ledger,
            event="bootstrap_forecast",
            payload={
                "artifact": str(args.artifact.resolve()),
                "artifact_sha256": expected_hash,
                "forecast": issued.as_dict(),
                "prior_manifest": str(args.prior_manifest.resolve()),
                "prior_manifest_sha256": sha256_file(args.prior_manifest),
            },
        )
    else:
        head = existing[-1]
        pending = head.get("payload", {}).get("forecast", {})
        if head.get("event") not in {"bootstrap_forecast", "release"}:
            raise RuntimeError("production ledger does not end with a pending forecast")
        if pending.get("target_date") != issued.target_date:
            raise RuntimeError("production ledger forecast does not match artifact")

    model.update_one(observation.week_start, observation.price_brl_per_liter)
    artifact_path = model.save(args.output_artifact)
    artifact_hash = sha256_file(artifact_path)
    restored = S10ProductionForecaster.load(
        artifact_path,
        expected_sha256=artifact_hash,
    )
    following = restored.predict_next()
    if following.as_dict() != model.predict_next().as_dict():
        raise RuntimeError("release serialization round-trip changed the forecast")
    error = observation.price_brl_per_liter - issued.point
    interval_hit = issued.p10 <= observation.price_brl_per_liter <= issued.p90
    outcome = append_audit_record(
        args.ledger,
        event="official_observation",
        payload={
            "observation": observation.as_dict(),
            "issued_forecast": issued.as_dict(),
            "error_brl_per_liter": error,
            "absolute_error_brl_per_liter": abs(error),
            "interval_hit": interval_hit,
            "source_artifact_sha256": expected_hash,
        },
    )
    release_payload: dict[str, object] = {
        "release_contract_version": "1.0",
        "release_status": "validated_candidate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_hash,
        "parent_artifact_sha256": expected_hash,
        "forecast": following.as_dict(),
        "health": restored.health().as_dict(),
        "metadata": restored.metadata(),
        "source_observation": observation.as_dict(),
        "outcome": {
            "forecast_target_date": issued.target_date,
            "actual": observation.price_brl_per_liter,
            "point": issued.point,
            "error_brl_per_liter": error,
            "absolute_error_brl_per_liter": abs(error),
            "interval_hit": interval_hit,
        },
        "quality_gates": {
            "artifact_integrity_verified": True,
            "source_schema_validated": True,
            "source_target_matches_pending_forecast": True,
            "serialization_roundtrip_exact": True,
            "forecast_finite": all(
                value == value and abs(value) != float("inf")
                for value in (following.point, following.p10, following.p90)
            ),
            "interval_ordered": 0 < following.p10 <= following.point <= following.p90,
            "fallback_used": following.fallback_used,
        },
        "audit_ledger": str(args.ledger.resolve()),
        "outcome_record_hash": outcome["record_hash"],
    }
    release = append_audit_record(
        args.ledger,
        event="release",
        payload={
            "artifact": release_payload["artifact"],
            "artifact_sha256": artifact_hash,
            "forecast": following.as_dict(),
            "parent_artifact_sha256": expected_hash,
        },
    )
    records = verify_audit_ledger(args.ledger)
    release_payload["release_record_hash"] = release["record_hash"]
    release_payload["ledger_records"] = len(records)
    release_payload["ledger_head_hash"] = records[-1]["record_hash"]
    _write_json(args.release_manifest, release_payload)
    if args.output is not None:
        _write_json(args.output, release_payload)
    return release_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workbook", type=Path, required=True)
    parser.add_argument("--source-url")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "s10_production.joblib",
    )
    parser.add_argument(
        "--prior-manifest",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_production" / "forecast.json",
    )
    parser.add_argument("--output-artifact", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_production" / "production_ledger.jsonl",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

