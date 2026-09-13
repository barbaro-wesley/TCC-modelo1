"""Intervalo condicional sobre o previsor pontual primario (ARIMA).

Este script NAO altera a previsao pontual.  Ele troca apenas a construcao do
intervalo: em vez de quantis de largura fixa sobre uma janela movel, usa uma
escala que responde a volatilidade recente e ao ultimo movimento do custo.

Protocolo: a especificacao da escala e fixada nos folds de desenvolvimento; o
holdout e lido ao final para reportar cobertura e largura.

Uso::

    python scripts/20_s10_conditional_interval.py --skip-holdout
    python scripts/20_s10_conditional_interval.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast  # noqa: E402
from vs_epl_krls.conditional_interval import ConditionalIntervalCalibrator  # noqa: E402
from vs_epl_krls.passthrough import build_passthrough_panel  # noqa: E402

HOLDOUT_SIZE = 104
VALIDATION_SIZE = 52
N_FOLDS = 3
QUIET_THRESHOLD = 0.02
ARIMA_START = 200
CACHE = ROOT / "reports" / "vs_epl_krls" / "s10_passthrough" / "arima_walkforward.csv"


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["data"])
    renamed = frame.rename(columns={"data": "date", "revenda": "price"})
    return build_passthrough_panel(renamed[["date", "price", "brent", "usdbrl"]])


def arima_walk_forward(panel: pd.DataFrame, *, refit_every: int = 13) -> np.ndarray:
    """Previsao ARIMA de um passo para todo o painel a partir de ``ARIMA_START``."""
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        if len(cached) == len(panel):
            return cached["arima"].to_numpy(float)
    price = panel["price"].to_numpy(float)
    predictions = np.full(len(panel), np.nan)
    model, last_fit = None, -(10**9)
    for index in range(ARIMA_START, len(panel)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None or index - last_fit >= refit_every:
                forecast, model = arima_forecast(price[:index], steps=1, model=None)
                last_fit = index
            else:
                forecast, model = arima_forecast(price[:index], steps=1, model=model)
        predictions[index] = float(forecast[-1])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": panel["date"], "arima": predictions}).to_csv(CACHE, index=False)
    return predictions


def moving_window_band(
    residuals: np.ndarray, point: float, *, window: int = 156, nominal: float = 0.80
) -> tuple[float, float, float]:
    """Intervalo incumbente: quantis de janela movel, largura independente do estado."""
    recent = residuals[-window:]
    recent = recent[np.isfinite(recent)]
    tail = (1.0 - nominal) / 2.0
    if recent.size < 20:
        return float("nan"), float("nan"), float("nan")
    low, high = np.quantile(recent, [tail, 1.0 - tail])
    return float(point + low), float(point + high), float(high - low)


def evaluate(panel: pd.DataFrame, arima: np.ndarray, start: int, end: int) -> pd.DataFrame:
    actual = panel["price"].to_numpy(float)
    records = []
    for index in range(start, end):
        point = arima[index]
        if not np.isfinite(point):
            continue
        history = np.arange(ARIMA_START, index)
        residual_history = actual[history] - arima[history]
        scale_history = panel.iloc[history]
        if residual_history.size < 60:
            continue
        calibrator = ConditionalIntervalCalibrator()
        try:
            calibrator.fit(residual_history, scale_history)
        except ValueError:
            continue
        band = calibrator.band(point, panel.iloc[index])
        fixed_low, fixed_high, fixed_width = moving_window_band(residual_history, point)
        records.append(
            {
                "date": panel["date"].iloc[index],
                "actual": actual[index],
                "persistence": panel["origin_price"].iloc[index],
                "arima": point,
                "cond_lower": band.lower,
                "cond_upper": band.upper,
                "cond_sigma": band.sigma,
                "fixed_lower": fixed_low,
                "fixed_upper": fixed_high,
            }
        )
    return pd.DataFrame.from_records(records)


def interval_score(actual: pd.Series, low: pd.Series, high: pd.Series, *, nominal: float) -> pd.Series:
    """Winkler / interval score: regra propria que pune largura e nao-cobertura.

    Menor e melhor.  Comparar apenas largura ou apenas cobertura escolhe o
    intervalo errado, porque um degenerado vence em cada uma isoladamente.
    """
    alpha = 1.0 - nominal
    below = (low - actual).clip(lower=0.0)
    above = (actual - high).clip(lower=0.0)
    return (high - low) + (2.0 / alpha) * (below + above)


def coverage_report(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    low, high = frame[f"{prefix}_lower"], frame[f"{prefix}_upper"]
    valid = low.notna() & high.notna()
    subset = frame[valid]
    inside = (subset["actual"] >= low[valid]) & (subset["actual"] <= high[valid])
    width = high[valid] - low[valid]
    quiet = (subset["actual"] - subset["persistence"]).abs() <= QUIET_THRESHOLD
    score = interval_score(subset["actual"], low[valid], high[valid], nominal=0.80)
    return {
        "interval_score": float(score.mean()),
        "interval_score_quiet": float(score[quiet].mean()) if quiet.any() else float("nan"),
        "interval_score_event": float(score[~quiet].mean()) if (~quiet).any() else float("nan"),
        "coverage": float(inside.mean()),
        "mean_width": float(width.mean()),
        "coverage_quiet": float(inside[quiet].mean()) if quiet.any() else float("nan"),
        "coverage_event": float(inside[~quiet].mean()) if (~quiet).any() else float("nan"),
        "width_quiet": float(width[quiet].mean()) if quiet.any() else float("nan"),
        "width_event": float(width[~quiet].mean()) if (~quiet).any() else float("nan"),
        "width_ratio_event_over_quiet": (
            float(width[~quiet].mean() / width[quiet].mean())
            if quiet.any() and (~quiet).any()
            else float("nan")
        ),
        "n": int(valid.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "processed" / "semanal_s10_features.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_interval"
    )
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.data)
    development_end = len(panel) - HOLDOUT_SIZE
    first = development_end - N_FOLDS * VALIDATION_SIZE
    print("calculando ARIMA walk-forward (cache)...")
    arima = arima_walk_forward(panel)

    development = evaluate(panel, arima, first, development_end)
    result = {
        "development": {
            "conditional": coverage_report(development, "cond"),
            "fixed_window": coverage_report(development, "fixed"),
        }
    }
    print("\n=== desenvolvimento (nominal 80%) ===")
    for name, row in result["development"].items():
        print(f"{name:16s} {json.dumps(row)}")
    development.to_csv(args.output_dir / "development_intervals.csv", index=False)

    if args.skip_holdout:
        print("\nholdout NAO lido.")
        (args.output_dir / "manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    holdout = evaluate(panel, arima, development_end, len(panel))
    result["holdout"] = {
        "conditional": coverage_report(holdout, "cond"),
        "fixed_window": coverage_report(holdout, "fixed"),
    }
    print("\n=== holdout final (nominal 80%) ===")
    for name, row in result["holdout"].items():
        print(f"{name:16s} {json.dumps(row)}")
    holdout.to_csv(args.output_dir / "holdout_intervals.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nartefatos em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
