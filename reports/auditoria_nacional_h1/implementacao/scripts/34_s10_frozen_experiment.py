"""Freeze and replay the TCC national h=1 development protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.experiment import (
    encoded,
    evaluate_fold,
    freeze_experiment,
    load_frozen,
    summarize,
)
from vs_epl_krls.fuel import load_anp_fuel_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, default=ROOT / "configs/s10_nacional_h1.json")
    freeze.add_argument("--data", type=Path, default=ROOT / "data/raw/anp_semanal_desde_2013.xlsx")
    freeze.add_argument(
        "--source-commit",
        required=True,
        help="Commit independently checked with git rev-parse HEAD",
    )
    freeze.add_argument("--repository-label", default=str(ROOT))
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--frozen", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--fold", help="Optional single frozen development fold")
    evaluate.add_argument(
        "--smoke-weeks", type=int, help="Small verification run; never treated as final evidence"
    )
    args = parser.parse_args()
    if args.command == "freeze":
        protocol = json.loads(args.config.read_text(encoding="utf-8"))
        history = load_anp_fuel_csv(args.data, products=["S10"], weekly="mean")
        result = freeze_experiment(
            ROOT,
            protocol,
            history,
            args.data,
            args.output,
            source_commit=args.source_commit,
            repository_label=args.repository_label,
        )
    else:
        manifest, history = load_frozen(ROOT, args.frozen)
        protocol = manifest["protocol"]
        folds = [fold for fold in protocol["folds"] if args.fold is None or fold["id"] == args.fold]
        if not folds:
            parser.error("unknown development fold")
        args.output.mkdir(parents=True, exist_ok=False)
        status = {
            "experiment_id": manifest["experiment_id"],
            "status": "running",
            "scope": "smoke_only" if args.smoke_weeks is not None else "development",
            "holdout_evaluated": False,
            "folds": [fold["id"] for fold in folds],
        }
        (args.output / "status.json").write_bytes(encoded(status))
        try:
            rows = []
            for fold in folds:
                rows.extend(evaluate_fold(history, protocol, fold, smoke_weeks=args.smoke_weeks))
            # Catch source/environment changes during execution before publishing results.
            load_frozen(ROOT, args.frozen)
            result = {
                **status,
                "status": "complete",
                "metrics": summarize(rows),
                "model_selection_performed": False,
                "production_promoted": False,
            }
            (args.output / "predictions.json").write_bytes(encoded(rows))
            (args.output / "result.json").write_bytes(encoded(result))
            (args.output / "status.json").write_bytes(encoded(result))
        except Exception as exc:
            (args.output / "status.json").write_bytes(
                encoded({**status, "status": "failed", "error": str(exc)})
            )
            raise
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
