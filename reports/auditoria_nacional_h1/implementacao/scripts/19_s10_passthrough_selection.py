"""Selecao e avaliacao do modelo de repasse (pass-through) para o S10.

Protocolo, identico em disciplina ao dos scripts 05 e 08:

1. a especificacao e os hiperparametros sao escolhidos apenas nos tres folds
   expansivos de desenvolvimento;
2. o holdout final de 104 semanas e lido **uma unica vez**, ao final, para
   reportar desempenho fora da amostra;
3. nenhum atributo usa informacao posterior a origem da previsao.

Uso::

    python scripts/19_s10_passthrough_selection.py
    python scripts/19_s10_passthrough_selection.py --skip-holdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast  # noqa: E402
from eval.metrics import diebold_mariano  # noqa: E402
from vs_epl_krls.passthrough import (  # noqa: E402
    PassThroughConfig,
    PassThroughECM,
    build_passthrough_panel,
)
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402
from vs_epl_krls.selection import pinned_validation_folds  # noqa: E402

HOLDOUT_SIZE = 104
VALIDATION_SIZE = 52
N_FOLDS = 3
QUIET_THRESHOLD = 0.02


def load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["data"])
    renamed = frame.rename(columns={"data": "date", "revenda": "price"})
    return build_passthrough_panel(renamed[["date", "price", "brent", "usdbrl"]])


def temporal_windows(
    dates: pd.Series,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Resolve folds, holdout e cauda prospectiva pelas datas congeladas.

    Ancorar o corte em ``len(panel)`` movia as tres janelas a cada semana nova
    publicada pela ANP, o que tornava a evidencia irreproduzivel e reabria o
    holdout sem aviso.
    """

    pinned = pinned_validation_folds(
        dates,
        validation_size=VALIDATION_SIZE,
        n_folds=N_FOLDS,
        expected_holdout_size=HOLDOUT_SIZE,
    )
    folds = [(fold.validation_start, fold.validation_end) for fold in pinned.folds]
    holdout = (pinned.holdout.validation_start, pinned.holdout.validation_end)
    prospective = (pinned.prospective.validation_start, pinned.prospective.validation_end)
    return folds, holdout, prospective


def arima_window(panel: pd.DataFrame, start: int, end: int, *, refit_every: int = 13):
    """ARIMA no nivel, mesma cadencia de refit usada no script 05."""
    price = panel["price"].to_numpy(float)
    predictions = np.full(end - start, np.nan)
    model, last_fit = None, -(10**9)
    for offset, index in enumerate(range(start, end)):
        history = price[:index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None or index - last_fit >= refit_every:
                forecast, model = arima_forecast(history, steps=1, model=None)
                last_fit = index
            else:
                forecast, model = arima_forecast(history, steps=1, model=model)
        predictions[offset] = float(forecast[-1])
    return predictions


def score(actual: np.ndarray, prediction: np.ndarray, naive: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(actual) & np.isfinite(naive)
    error = actual[valid] - prediction[valid]
    naive_error = actual[valid] - naive[valid]
    quiet = np.abs(actual[valid] - naive[valid]) <= QUIET_THRESHOLD
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "median_abs_error": float(np.median(np.abs(error))),
        "rmse_quiet": float(np.sqrt(np.mean(error[quiet] ** 2))) if quiet.any() else float("nan"),
        "rmse_event": (
            float(np.sqrt(np.mean(error[~quiet] ** 2))) if (~quiet).any() else float("nan")
        ),
        "rmse_ratio_vs_naive": float(
            np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(naive_error**2))
        ),
        "n": int(valid.sum()),
    }


def evaluate_window(
    panel: pd.DataFrame, start: int, end: int, config: PassThroughConfig
) -> pd.DataFrame:
    model = PassThroughECM(config=config)
    frame = model.walk_forward(panel, start, end, refit_every=1)
    frame["arima"] = arima_window(panel, start, end)
    frame["target_date"] = frame["date"]
    return frame


def block_bootstrap_pvalue(
    error_a: np.ndarray, error_b: np.ndarray, *, block: int = 8, draws: int = 4000, seed: int = 42
) -> float:
    """Teste bilateral por bootstrap em blocos sobre a diferenca de erro quadratico.

    Com o erro concentrado em poucas semanas, o Diebold-Mariano assintotico
    tem pouco poder; o bootstrap em blocos preserva a dependencia temporal.
    """

    loss = error_a**2 - error_b**2
    loss = loss[np.isfinite(loss)]
    if loss.size < 16:
        return float("nan")
    rng = np.random.default_rng(seed)
    centered = loss - loss.mean()
    starts = np.arange(loss.size - block + 1)
    needed = int(np.ceil(loss.size / block))
    observed = loss.mean()
    count = 0
    for _ in range(draws):
        picks = rng.choice(starts, size=needed)
        sample = np.concatenate([centered[s : s + block] for s in picks])[: loss.size]
        if abs(sample.mean()) >= abs(observed):
            count += 1
    return float((count + 1) / (draws + 1))


def coverage(frame: pd.DataFrame) -> dict[str, float]:
    valid = frame[["actual", "lower", "upper"]].notna().all(axis=1)
    subset = frame[valid]
    if subset.empty:
        return {"coverage": float("nan"), "mean_width": float("nan")}
    inside = (subset["actual"] >= subset["lower"]) & (subset["actual"] <= subset["upper"])
    quiet = (subset["actual"] - subset["persistence"]).abs() <= QUIET_THRESHOLD
    return {
        "coverage": float(inside.mean()),
        "mean_width": float((subset["upper"] - subset["lower"]).mean()),
        "mean_width_quiet": float((subset["upper"] - subset["lower"])[quiet].mean())
        if quiet.any()
        else float("nan"),
        "mean_width_event": float((subset["upper"] - subset["lower"])[~quiet].mean())
        if (~quiet).any()
        else float("nan"),
        "coverage_quiet": float(inside[quiet].mean()) if quiet.any() else float("nan"),
        "coverage_event": float(inside[~quiet].mean()) if (~quiet).any() else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "processed" / "semanal_s10_features.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_passthrough",
    )
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(args.data)
    folds, holdout, prospective = temporal_windows(panel["date"])
    config = PassThroughConfig()
    started = time.perf_counter()

    print(f"painel: {len(panel)} semanas, {panel['date'].min().date()} a {panel['date'].max().date()}")
    print(
        f"janela congelada: holdout {panel['date'][holdout[0]].date()} a "
        f"{panel['date'][holdout[1] - 1].date()}; "
        f"{prospective[1] - prospective[0]} semana(s) prospectiva(s) fora do protocolo"
    )
    print("folds de desenvolvimento:")
    development_rows: list[dict[str, object]] = []
    development_frames: list[pd.DataFrame] = []
    for index, (start, end) in enumerate(folds, start=1):
        frame = evaluate_window(panel, start, end, config)
        development_frames.append(frame)
        actual = frame["actual"].to_numpy(float)
        naive = frame["persistence"].to_numpy(float)
        row_pt = score(actual, frame["prediction"].to_numpy(float), naive)
        row_ar = score(actual, frame["arima"].to_numpy(float), naive)
        row_na = score(actual, naive, naive)
        print(
            f"  fold {index} {frame['date'].min().date()}..{frame['date'].max().date()} "
            f"| pass-through {row_pt['rmse']:.5f} | ARIMA {row_ar['rmse']:.5f} "
            f"| persistencia {row_na['rmse']:.5f}"
        )
        for name, row in (("pass_through", row_pt), ("arima", row_ar), ("persistencia", row_na)):
            development_rows.append({"fold": f"validation_{index}", "model": name, **row})

    development = pd.DataFrame(development_rows)
    summary = (
        development.groupby("model")[["rmse", "mae", "rmse_ratio_vs_naive"]]
        .agg(["mean", "max"])
        .round(6)
    )
    print("\n=== resumo de desenvolvimento ===")
    print(summary.to_string())

    development_all = pd.concat(development_frames, ignore_index=True)
    development_coverage = coverage(development_all)
    print(f"\ncobertura do intervalo no desenvolvimento: {development_coverage}")

    pt_mean = development[development.model == "pass_through"]["rmse"].mean()
    ar_mean = development[development.model == "arima"]["rmse"].mean()
    pt_worst = development[development.model == "pass_through"]["rmse_ratio_vs_naive"].max()
    gates = {
        "beats_arima_on_development_mean": bool(pt_mean < ar_mean),
        "beats_naive_every_development_fold": bool(pt_worst < 1.0),
        "development_gain_vs_arima_pct": float(100.0 * (1.0 - pt_mean / ar_mean)),
        "interval_coverage_within_band": bool(
            0.70 <= development_coverage["coverage"] <= 0.95
        ),
    }
    gates["cleared_for_holdout_read"] = bool(
        gates["beats_arima_on_development_mean"]
        and gates["beats_naive_every_development_fold"]
        and gates["interval_coverage_within_band"]
    )
    print(f"\ngates de desenvolvimento: {json.dumps(gates, indent=2)}")

    development.to_csv(args.output_dir / "development_folds.csv", index=False)
    development_all.to_csv(args.output_dir / "development_predictions.csv", index=False)

    manifest: dict[str, object] = {
        "scope": "ANP weekly national resale price — Diesel B S10 only",
        "horizon_weeks": 1,
        "protocol": "spec frozen on three expanding development folds; holdout read once",
        "data_start": str(panel["date"].min().date()),
        "data_end": str(panel["date"].max().date()),
        "n_observations": int(len(panel)),
        "window_pinned_by": "calendar dates, not panel length",
        "prospective_weeks_outside_protocol": int(prospective[1] - prospective[0]),
        "development_folds": [
            {"start": str(panel['date'][a].date()), "end": str(panel['date'][b - 1].date())}
            for a, b in folds
        ],
        "development_summary": json.loads(development.to_json(orient="records")),
        "development_coverage": development_coverage,
        "development_gates": gates,
        "config": config.__dict__,
        "seconds": round(time.perf_counter() - started, 2),
    }

    if args.skip_holdout:
        print("\nholdout NAO foi lido (--skip-holdout).")
        manifest["holdout_evaluated"] = False
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return 0

    if not gates["cleared_for_holdout_read"]:
        print("\ngates de desenvolvimento reprovados: o holdout permanece fechado.")
        manifest["holdout_evaluated"] = False
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return 1

    print("\n=== leitura unica do holdout final ===")
    start, end = holdout
    frame = evaluate_window(panel, start, end, config)
    actual = frame["actual"].to_numpy(float)
    naive = frame["persistence"].to_numpy(float)
    rows = {
        "pass_through": score(actual, frame["prediction"].to_numpy(float), naive),
        "arima": score(actual, frame["arima"].to_numpy(float), naive),
        "persistencia": score(actual, naive, naive),
    }
    error_pt = actual - frame["prediction"].to_numpy(float)
    error_ar = actual - frame["arima"].to_numpy(float)
    error_na = actual - naive
    comparison = pd.DataFrame(
        [{"model": name, **row} for name, row in rows.items()]
    ).sort_values("rmse", ignore_index=True)
    print(comparison.to_string(index=False))

    tests = {
        "dm_pvalue_pt_vs_naive": float(
            diebold_mariano(error_pt, error_na, h=1)["pvalue"]
        ),
        "dm_pvalue_pt_vs_arima": float(
            diebold_mariano(error_pt, error_ar, h=1)["pvalue"]
        ),
        "block_bootstrap_pvalue_pt_vs_naive": block_bootstrap_pvalue(error_pt, error_na),
        "block_bootstrap_pvalue_pt_vs_arima": block_bootstrap_pvalue(error_pt, error_ar),
    }
    print(f"\ntestes: {json.dumps(tests, indent=2)}")

    holdout_coverage = coverage(frame)
    print(f"cobertura no holdout: {json.dumps(holdout_coverage, indent=2)}")

    procurement = {}
    for column, name in (("prediction", "pass_through"), ("arima", "ARIMA")):
        try:
            backtest = simulate_one_week_prebuy(
                frame[["target_date", "actual", "persistence", column]],
                prediction_column=column,
                model_name=name,
            )
            procurement[name] = {
                "net_savings_brl": backtest.net_savings_brl,
                "annualized_savings_brl": backtest.annualized_savings_brl,
                "annualized_savings_ci90_brl": list(backtest.annualized_savings_ci90_brl),
                "triggered_prebuys": backtest.triggered_prebuys,
                "trigger_precision": backtest.trigger_precision,
                "largest_event_share_of_savings": backtest.largest_event_share_of_savings,
            }
        except ValueError as exc:
            procurement[name] = {"error": str(exc)}
    print(f"\npolitica de compra: {json.dumps(procurement, indent=2, default=str)}")

    frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    comparison.to_csv(args.output_dir / "holdout_comparison.csv", index=False)
    manifest.update(
        {
            "holdout_evaluated": True,
            "holdout_window": {
                "start": str(panel["date"][start].date()),
                "end": str(panel["date"][end - 1].date()),
            },
            "holdout_comparison": json.loads(comparison.to_json(orient="records")),
            "holdout_tests": tests,
            "holdout_coverage": holdout_coverage,
            "procurement": procurement,
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nartefatos em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
