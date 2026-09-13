"""Replay the prespecified S10 one-week prebuy policy on the frozen holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.procurement import simulate_one_week_prebuy


def run(args: argparse.Namespace) -> dict[str, object]:
    predictions = pd.read_csv(args.predictions)
    result = simulate_one_week_prebuy(
        predictions,
        prediction_column=args.prediction_column,
        model_name=args.model_name,
        monthly_liters=args.monthly_liters,
        flexibility_fraction=args.flexibility_fraction,
        signal_threshold_brl_per_liter=args.signal_threshold,
        carrying_cost_brl_per_liter_week=args.carrying_cost,
        random_state=args.random_state,
    )
    payload = result.as_dict()
    payload["methodology"] = {
        "policy": "buy a fixed share of next week's demand one week early only when the frozen forecast exceeds the known price by the fixed threshold",
        "causal_alignment": "each decision uses only the prediction and price available at its origin",
        "not_included": [
            "supplier discounts",
            "tax differences",
            "storage constraints beyond the configured one-week share",
            "financing cost unless supplied explicitly",
        ],
        "claim_boundary": "historical policy replay, not guaranteed future savings",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_selection" / "holdout_predictions_h1.csv",
    )
    parser.add_argument("--prediction-column", default="arima")
    parser.add_argument("--model-name", default="ARIMA")
    parser.add_argument("--monthly-liters", type=float, default=200_000.0)
    parser.add_argument("--flexibility-fraction", type=float, default=0.25)
    parser.add_argument("--signal-threshold", type=float, default=0.01)
    parser.add_argument("--carrying-cost", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_product" / "procurement_backtest.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

