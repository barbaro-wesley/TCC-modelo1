"""Build a self-contained, evidence-backed investor JSON snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.anp_official import sha256_file
from vs_epl_krls.product import S10ProductService


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    service = S10ProductService(
        args.artifact,
        args.manifest,
        selection_manifest=args.selection_manifest,
        procurement_report=args.procurement_report,
    )
    load_evidence = (
        json.loads(args.load_report.read_text(encoding="utf-8"))
        if args.load_report.exists()
        else None
    )
    quality_evidence = (
        json.loads(args.quality_summary.read_text(encoding="utf-8"))
        if args.quality_summary.exists()
        else None
    )
    supply_chain_evidence = (
        {
            "sbom_path": str(args.sbom.resolve()),
            "sbom_sha256": sha256_file(args.sbom),
        }
        if args.sbom.exists()
        else None
    )
    snapshot: dict[str, object] = {
        "product": "S10 Intelligence",
        "positioning": "weekly decision support for Brazilian Diesel B S10 procurement",
        "forecast": service.forecast(),
        "models": service.model_catalog(),
        "model_evidence": service.model_evidence(),
        "procurement_evidence": service.procurement_evidence(),
        "load_evidence": load_evidence,
        "quality_evidence": quality_evidence,
        "supply_chain_evidence": supply_chain_evidence,
        "claim_boundaries": [
            "national ANP average, not a supplier quote",
            "historical policy replay is not guaranteed future savings",
            "ARIMA remains primary; VS-ePL-KRLS is a non-promoted challenger",
            "one prospective outcome is insufficient for challenger promotion",
        ],
    }
    _atomic_text(
        args.snapshot,
        json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False),
    )
    return {
        "snapshot": str(args.snapshot.resolve()),
        "status": service.status().as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=ROOT / "artifacts" / "releases" / "s10_production_2026-08-16.joblib",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "releases" / "2026-08-16.json",
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_selection" / "selection_manifest_h1.json",
    )
    parser.add_argument(
        "--procurement-report", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "procurement_backtest.json",
    )
    parser.add_argument(
        "--load-report", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "load_smoke.json",
    )
    parser.add_argument(
        "--quality-summary", type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_product"
        / "quality_summary.json",
    )
    parser.add_argument(
        "--sbom", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "sbom.cdx.json",
    )
    parser.add_argument(
        "--snapshot", type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "investor_snapshot.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
