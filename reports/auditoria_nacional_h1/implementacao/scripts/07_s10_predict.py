"""Operate the versioned one-week Diesel B S10 production artifact.

Without update arguments this command is read-only.  Supplying both
``--update-date`` and ``--update-price`` consumes the newly published ANP week,
refits the compact classical components, updates the VS challenger online and
atomically saves a new artifact before producing the following forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.production import S10ProductionForecaster


def run(args: argparse.Namespace) -> dict[str, object]:
    if (args.update_date is None) != (args.update_price is None):
        raise ValueError("--update-date and --update-price must be supplied together")
    model = S10ProductionForecaster.load(
        args.artifact,
        expected_sha256=args.expected_sha256,
    )
    updated = args.update_date is not None
    saved_artifact: str | None = None
    if updated:
        model.update_one(
            args.update_date,
            args.update_price,
            allow_anomalous_change=args.allow_anomalous_change,
        )
        destination = args.output_artifact or args.artifact
        saved_artifact = str(model.save(destination).resolve())
    payload: dict[str, object] = {
        "forecast": model.predict_next().as_dict(),
        "health": model.health().as_dict(),
        "metadata": model.metadata(),
        "updated": updated,
        "saved_artifact": saved_artifact,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        temporary.replace(args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "s10_production.joblib",
    )
    parser.add_argument(
        "--expected-sha256",
        help="hash esperado, verificado antes da desserialização do joblib",
    )
    parser.add_argument("--update-date", help="data da nova observação ANP")
    parser.add_argument("--update-price", type=float, help="preço médio S10 observado")
    parser.add_argument(
        "--allow-anomalous-change",
        action="store_true",
        help="aceita mudança fora do limite robusto após revisão humana",
    )
    parser.add_argument(
        "--output-artifact",
        type=Path,
        help="destino opcional; por padrão atualiza o artefato de entrada atomicamente",
    )
    parser.add_argument("--output", type=Path, help="JSON operacional opcional")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
