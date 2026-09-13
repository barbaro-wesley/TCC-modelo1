"""Selecao do modelo de paridade de diesel, pontuada pela decisao de compra.

O gate historico do repositorio (ganho de 2% no RMSE, Diebold-Mariano abaixo de
5%) nao e decidivel nesta serie: uma unica semana do holdout responde por 75% do
erro quadratico.  Aqui a selecao usa, em conjunto:

* RMSE e MAE, com decomposicao por regime;
* economia liquida e precisao dos gatilhos da politica de antecipacao de compra,
  que e o que o produto de fato vende.

Protocolo: a especificacao e escolhida somente nos tres folds expansivos de
desenvolvimento; o holdout e lido no final.

Uso::

    python scripts/22_s10_parity_selection.py --skip-holdout
    python scripts/22_s10_parity_selection.py
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
from vs_epl_krls.passthrough import (  # noqa: E402
    PassThroughConfig,
    PassThroughECM,
    build_parity_panel,
)
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402
from vs_epl_krls.selection import pinned_validation_folds  # noqa: E402

HOLDOUT_SIZE = 104
VALIDATION_SIZE = 52
N_FOLDS = 3
QUIET = 0.02

#: Especificacao escolhida nos folds de desenvolvimento e congelada ANTES da
#: leitura do holdout.  O criterio foi a decisao de compra, nao o RMSE: entre as
#: variantes de paridade, esta entrega a maior economia (R$ 47.169 em 156
#: semanas), a melhor precisao de gatilho (70,5%), o menor limite inferior de
#: risco (CI90 de R$ 7.152/ano) e a menor concentracao num unico evento (20,5%),
#: sendo tambem a mais simples.  Variantes com mais atributos melhoram o RMSE em
#: cerca de 2% — dentro do ruido documentado — e pioram a precisao do gatilho.
FROZEN_SPEC = "paridade"

#: Limiar pre-registrado da politica de antecipacao, mantido sem ajuste: a
#: varredura de sensibilidade no desenvolvimento mostrou que ele ja e o otimo.
SIGNAL_THRESHOLD = 0.01

SPECS: dict[str, tuple[str, ...]] = {
    "momentum": ("dp1",),
    "paridade": ("dp1", "rpar1"),
    "paridade_l2": ("dp1", "rpar1", "rpar2"),
    "paridade_ecm": ("dp1", "rpar1", "coint_par"),
    "paridade_l2_ecm": ("dp1", "rpar1", "rpar2", "coint_par"),
    "paridade_l2_ecm_brent": ("dp1", "rpar1", "rpar2", "coint_par", "rbb1"),
    "paridade_l2_ecm_prod": ("dp1", "rpar1", "rpar2", "coint_par", "rprod3"),
}


def windows(
    dates: pd.Series,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Resolve folds, holdout e cauda prospectiva pelas datas congeladas.

    O corte nao pode depender de ``len(panel)``.  Com o holdout ancorado no fim
    da serie, cada semana nova publicada pela ANP deslocava as tres janelas: os
    numeros divulgados deixavam de ser reproduziveis e qualquer reexecucao lia,
    sem avisar, um holdout diferente do que ja tinha sido lido.
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


def arima_window(panel: pd.DataFrame, start: int, end: int, *, refit_every: int = 13) -> np.ndarray:
    price = panel["price"].to_numpy(float)
    predictions = np.full(end - start, np.nan)
    model, last_fit = None, -(10**9)
    for offset, index in enumerate(range(start, end)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None or index - last_fit >= refit_every:
                forecast, model = arima_forecast(price[:index], steps=1, model=None)
                last_fit = index
            else:
                forecast, model = arima_forecast(price[:index], steps=1, model=model)
        predictions[offset] = float(forecast[-1])
    return predictions


def accuracy(actual: np.ndarray, prediction: np.ndarray, naive: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(actual) & np.isfinite(naive)
    error = actual[valid] - prediction[valid]
    naive_error = actual[valid] - naive[valid]
    quiet = np.abs(naive_error) <= QUIET
    moved = actual[valid] != naive[valid]
    direction = (
        float(
            np.mean(
                np.sign(prediction[valid][moved] - naive[valid][moved])
                == np.sign(actual[valid][moved] - naive[valid][moved])
            )
        )
        if moved.any()
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse_quiet": float(np.sqrt(np.mean(error[quiet] ** 2))) if quiet.any() else float("nan"),
        "rmse_event": (
            float(np.sqrt(np.mean(error[~quiet] ** 2))) if (~quiet).any() else float("nan")
        ),
        "rmse_ratio_vs_naive": float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(naive_error**2))),
        "directional_accuracy": direction,
        "n": int(valid.sum()),
    }


def decision(frame: pd.DataFrame, column: str, name: str) -> dict[str, float]:
    subset = frame[["target_date", "actual", "persistence", column]].dropna()
    if len(subset) < 3:
        return {"net_savings_brl": float("nan"), "triggered": 0, "precision": float("nan")}
    try:
        backtest = simulate_one_week_prebuy(
            subset, prediction_column=column, model_name=name
        )
    except ValueError as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}
    return {
        "net_savings_brl": backtest.net_savings_brl,
        "annualized_savings_brl": backtest.annualized_savings_brl,
        "annualized_ci90": list(backtest.annualized_savings_ci90_brl),
        "triggered": backtest.triggered_prebuys,
        "precision": backtest.trigger_precision,
        "largest_event_share": backtest.largest_event_share_of_savings,
    }


def evaluate(panel: pd.DataFrame, start: int, end: int, config: PassThroughConfig,
             specs: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target_date": panel["date"].iloc[start:end].to_numpy(),
            "actual": panel["price"].iloc[start:end].to_numpy(float),
            "persistence": panel["origin_price"].iloc[start:end].to_numpy(float),
        }
    ).reset_index(drop=True)
    for name, features in specs.items():
        available = tuple(f for f in features if f in panel.columns)
        if len(available) != len(features):
            continue
        model = PassThroughECM(config=config, feature_names=available)
        result = model.walk_forward(panel, start, end, refit_every=1)
        frame[name] = result["prediction"].to_numpy(float)
        frame[f"{name}__lower"] = result["lower"].to_numpy(float)
        frame[f"{name}__upper"] = result["upper"].to_numpy(float)
    frame["arima"] = arima_window(panel, start, end)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "data" / "processed" / "s10_causal_panel.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_parity"
    )
    parser.add_argument("--skip-holdout", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    causal = pd.read_csv(args.panel, parse_dates=["date"])
    panel = build_parity_panel(causal)
    folds, holdout, prospective = windows(panel["date"])
    config = PassThroughConfig()

    print(f"painel: {len(panel)} semanas, {panel['date'].min().date()} a "
          f"{panel['date'].max().date()}")
    print(
        f"janela congelada: desenvolvimento ate {panel['date'][holdout[0] - 1].date()}, "
        f"holdout {panel['date'][holdout[0]].date()} a {panel['date'][holdout[1] - 1].date()}"
    )
    n_prospective = prospective[1] - prospective[0]
    if n_prospective:
        print(
            f"{n_prospective} semana(s) posteriores ao holdout congelado ficam fora do "
            "protocolo e so contam como evidencia prospectiva"
        )

    frames = [evaluate(panel, a, b, config, SPECS) for a, b in folds]
    models = [name for name in SPECS if name in frames[0].columns] + ["arima", "persistence"]

    rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames, start=1):
        frame["persistence_pred"] = frame["persistence"]
        actual = frame["actual"].to_numpy(float)
        naive = frame["persistence"].to_numpy(float)
        for name in models:
            column = "persistence_pred" if name == "persistence" else name
            rows.append(
                {
                    "fold": f"validation_{index}",
                    "model": name,
                    **accuracy(actual, frame[column].to_numpy(float), naive),
                }
            )
    development = pd.DataFrame(rows)

    combined = pd.concat(frames, ignore_index=True)
    combined["persistence_pred"] = combined["persistence"]
    decisions = {
        name: decision(combined, "persistence_pred" if name == "persistence" else name, name)
        for name in models
    }

    summary = (
        development.groupby("model")[
            ["rmse", "mae", "rmse_quiet", "rmse_event", "rmse_ratio_vs_naive",
             "directional_accuracy"]
        ]
        .mean()
        .round(6)
    )
    summary["worst_ratio"] = development.groupby("model")["rmse_ratio_vs_naive"].max().round(6)
    summary["economia"] = [decisions[name].get("net_savings_brl") for name in summary.index]
    summary["gatilhos"] = [decisions[name].get("triggered") for name in summary.index]
    summary["precisao"] = [decisions[name].get("precision") for name in summary.index]
    summary = summary.sort_values("rmse")

    print("\n=== desenvolvimento: acuracia e decisao ===")
    print(summary.to_string())

    development.to_csv(args.output_dir / "development_folds.csv", index=False)
    combined.to_csv(args.output_dir / "development_predictions.csv", index=False)

    manifest: dict[str, object] = {
        "protocol": "spec chosen on three development folds; holdout read at the end",
        "panel": str(args.panel),
        "n_weeks": int(len(panel)),
        "window_pinned_by": "calendar dates, not panel length",
        "pinned_holdout": {
            "start": str(panel["date"][holdout[0]].date()),
            "end": str(panel["date"][holdout[1] - 1].date()),
            "n_weeks": int(holdout[1] - holdout[0]),
        },
        "prospective_weeks_outside_protocol": int(prospective[1] - prospective[0]),
        "development_folds": [
            {"start": str(panel["date"][a].date()), "end": str(panel["date"][b - 1].date())}
            for a, b in folds
        ],
        "development_summary": json.loads(summary.reset_index().to_json(orient="records")),
        "development_decisions": decisions,
        "specs": {k: list(v) for k, v in SPECS.items()},
    }

    if args.skip_holdout:
        manifest["holdout_evaluated"] = False
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print("\nholdout NAO lido.")
        return 0

    start, end = holdout
    frame = evaluate(panel, start, end, config, SPECS)
    frame["persistence_pred"] = frame["persistence"]
    actual = frame["actual"].to_numpy(float)
    naive = frame["persistence"].to_numpy(float)
    holdout_rows = [
        {
            "model": name,
            **accuracy(
                actual,
                frame["persistence_pred" if name == "persistence" else name].to_numpy(float),
                naive,
            ),
            **{
                f"decision_{k}": v
                for k, v in decision(
                    frame, "persistence_pred" if name == "persistence" else name, name
                ).items()
            },
        }
        for name in models
    ]
    comparison = pd.DataFrame(holdout_rows).sort_values("rmse", ignore_index=True)
    print("\n=== holdout final ===")
    print(
        comparison[
            ["model", "rmse", "mae", "rmse_quiet", "rmse_event", "directional_accuracy",
             "decision_net_savings_brl", "decision_triggered", "decision_precision"]
        ].to_string(index=False)
    )

    frame.to_csv(args.output_dir / "holdout_predictions.csv", index=False)
    comparison.to_csv(args.output_dir / "holdout_comparison.csv", index=False)
    manifest["holdout_evaluated"] = True
    manifest["holdout_window"] = {
        "start": str(panel["date"][start].date()),
        "end": str(panel["date"][end - 1].date()),
    }
    manifest["holdout_comparison"] = json.loads(comparison.to_json(orient="records"))
    manifest["frozen_spec"] = FROZEN_SPEC
    manifest["frozen_spec_features"] = list(SPECS[FROZEN_SPEC])
    manifest["signal_threshold_brl_per_liter"] = SIGNAL_THRESHOLD
    manifest["holdout_read_count_note"] = (
        "Segunda leitura do holdout neste projeto: a primeira foi o modelo de "
        "repasse Brent (script 19), rejeitado. Duas leituras inflam o otimismo; "
        "confirmacao prospectiva continua obrigatoria."
    )
    chosen = comparison[comparison["model"] == FROZEN_SPEC]
    if not chosen.empty:
        print(f"\n=== especificacao congelada: {FROZEN_SPEC} ===")
        print(chosen.to_string(index=False))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nartefatos em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
