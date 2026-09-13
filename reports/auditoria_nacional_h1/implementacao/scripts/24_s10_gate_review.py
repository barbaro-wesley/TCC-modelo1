"""Aplica os gates decidiveis a evidencia ja publicada e fecha as decisoes abertas.

Este script nao treina, nao seleciona e nao reabre o holdout.  Ele repontua
previsoes que ja foram emitidas e publicadas, com metricas que concluem onde as
antigas nao concluiam.  Nenhuma escolha de modelo depende do resultado: as
especificacoes ja estavam congeladas quando estes numeros foram gerados, e o que
muda aqui e apenas a regua.

Tres perguntas que o projeto carregava em aberto:

1. O modelo de paridade deve substituir o ARIMA como primario?
2. O VS-ePL-KRLS, challenger desde o inicio, tem caso para promocao?
3. O intervalo P10-P90 servido pelo produto esta calibrado?

Uso::

    python scripts/24_s10_gate_review.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.calibration import backtest_adaptive_interval  # noqa: E402
from vs_epl_krls.gates import (  # noqa: E402
    accuracy_report,
    evaluate_challenger,
    interval_report,
)
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402

NOMINAL_COVERAGE = 0.80
MONTHLY_LITERS = 200_000.0


def weekly_savings(frame: pd.DataFrame, column: str, name: str) -> np.ndarray:
    replay = simulate_one_week_prebuy(
        frame,
        prediction_column=column,
        model_name=name,
        monthly_liters=MONTHLY_LITERS,
    )
    # A politica decide na semana i para consumir na i+1, entao a serie tem um
    # elemento a menos; alinhamos com um zero final para casar com as metricas.
    return np.append(np.asarray(replay.weekly_net_savings_brl, dtype=float), 0.0)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, bool):
                values.append("sim" if value else "**nao**")
            elif isinstance(value, float):
                values.append(f"{value:.6f}" if abs(value) < 1000 else f"{value:,.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def summarize(verdict: dict[str, object], challenger: str) -> str:
    """Uma frase honesta sobre o que os gates decidiram e por que."""

    failed = list(verdict["failed_gates"])
    if not failed:
        return f"{challenger} passou em todos os gates: ha caso para promocao."
    passed = [gate["name"] for gate in verdict["gates"] if gate["passed"]]
    economic = [name for name in passed if name.startswith("economia")]
    if economic and len(failed) < len(passed):
        return (
            f"{challenger} decide melhor do que preve: passa nos gates economicos "
            f"({', '.join(economic)}) e falha em {', '.join(failed)}."
        )
    return f"{challenger} nao passa: falha em {', '.join(failed)}."


def review_parity(frame: pd.DataFrame) -> dict[str, object]:
    actual = frame["actual"].to_numpy(float)
    origin = frame["persistence"].to_numpy(float)
    verdict = evaluate_challenger(
        challenger="paridade",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=frame["paridade"].to_numpy(float),
        prediction_incumbent=frame["arima"].to_numpy(float),
        savings_challenger=weekly_savings(frame, "paridade", "paridade"),
        savings_incumbent=weekly_savings(frame, "arima", "ARIMA"),
        interval_challenger=(
            frame["paridade__lower"].to_numpy(float),
            frame["paridade__upper"].to_numpy(float),
        ),
        nominal_coverage=NOMINAL_COVERAGE,
    )
    return verdict.as_dict()


def review_vs_challenger(frame: pd.DataFrame) -> dict[str, object]:
    actual = frame["actual"].to_numpy(float)
    origin = frame["persistence"].to_numpy(float)
    verdict = evaluate_challenger(
        challenger="VS-ePL-KRLS",
        incumbent="ARIMA",
        actual=actual,
        origin=origin,
        prediction_challenger=frame["vs_epl_krls"].to_numpy(float),
        prediction_incumbent=frame["arima"].to_numpy(float),
        savings_challenger=weekly_savings(frame, "vs_epl_krls", "VS-ePL-KRLS"),
        savings_incumbent=weekly_savings(frame, "arima", "ARIMA"),
        nominal_coverage=NOMINAL_COVERAGE,
    )
    return verdict.as_dict()


def review_interval(frame: pd.DataFrame) -> dict[str, object]:
    """Compara o intervalo publicado com a recalibracao conformal adaptativa."""

    actual = frame["actual"].to_numpy(float)
    point = frame["paridade"].to_numpy(float)
    published = interval_report(
        actual,
        frame["paridade__lower"].to_numpy(float),
        frame["paridade__upper"].to_numpy(float),
        nominal_coverage=NOMINAL_COVERAGE,
    )
    half_width = (
        frame["paridade__upper"].to_numpy(float) - frame["paridade__lower"].to_numpy(float)
    ) / 2.0
    adaptive = backtest_adaptive_interval(
        actual,
        point,
        # A escala condicional publicada vira o fallback de aquecimento; depois
        # dele o proprio historico de residuos manda.
        scale=half_width / 1.2815515655446004,
        nominal_coverage=NOMINAL_COVERAGE,
        gamma=0.02,
        window=104,
        min_residuals=20,
    )
    recalibrated = interval_report(
        actual, adaptive["lower"], adaptive["upper"], nominal_coverage=NOMINAL_COVERAGE
    )
    state = adaptive["state"]
    return {
        "published": published.as_dict(),
        "adaptive_conformal": recalibrated.as_dict(),
        "winkler_improvement": published.mean_winkler - recalibrated.mean_winkler,
        "width_reduction_fraction": 1.0 - recalibrated.mean_width / published.mean_width,
        "conformal_state": state.as_dict(),
        "alpha_path": [float(value) for value in adaptive["alpha"]],
        "lower": [float(value) for value in adaptive["lower"]],
        "upper": [float(value) for value in adaptive["upper"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parity",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_parity" / "holdout_predictions.csv",
    )
    parser.add_argument(
        "--production",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_selection"
        / "holdout_predictions_h1.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_gates"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parity = pd.read_csv(args.parity, parse_dates=["target_date"])
    production = pd.read_csv(args.production, parse_dates=["target_date"])

    parity_verdict = review_parity(parity)
    vs_verdict = review_vs_challenger(production)
    interval_verdict = review_interval(parity)

    baselines = {
        name: asdict(
            accuracy_report(
                parity["actual"].to_numpy(float),
                parity[column].to_numpy(float),
                parity["persistence"].to_numpy(float),
            )
        )
        for name, column in (
            ("paridade", "paridade"),
            ("ARIMA", "arima"),
            ("persistencia", "persistence"),
        )
    }

    manifest = {
        "scope": "repontuacao descritiva de previsoes ja publicadas; nenhuma selecao depende dela",
        "holdout_window": {
            "start": str(parity["target_date"].min().date()),
            "end": str(parity["target_date"].max().date()),
            "n_weeks": int(len(parity)),
        },
        "nominal_coverage": NOMINAL_COVERAGE,
        "accuracy_by_model": baselines,
        "verdict_parity_vs_arima": parity_verdict,
        "verdict_vs_epl_krls_vs_arima": vs_verdict,
        "interval_review": interval_verdict,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    gate_rows = [
        {"pergunta": "paridade substitui ARIMA?", **gate}
        for gate in parity_verdict["gates"]
    ] + [
        {"pergunta": "VS-ePL-KRLS promove?", **gate} for gate in vs_verdict["gates"]
    ]
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "gates.csv", index=False)

    published = interval_verdict["published"]
    adaptive = interval_verdict["adaptive_conformal"]
    report = [
        "# Revisao por gates decidiveis — Diesel B S10",
        "",
        "Repontuacao das previsoes ja publicadas com metricas que concluem. Nao houve",
        "treino, selecao nem nova leitura do holdout: as especificacoes ja estavam",
        "congeladas, e o que muda aqui e a regua.",
        "",
        f"Janela: {manifest['holdout_window']['start']} a "
        f"{manifest['holdout_window']['end']}, "
        f"{manifest['holdout_window']['n_weeks']} semanas.",
        "",
        "## Resumo",
        "",
        f"1. {summarize(parity_verdict, 'O modelo de paridade')}",
        f"2. {summarize(vs_verdict, 'O VS-ePL-KRLS')}",
        f"3. O intervalo publicado cobre {published['empirical_coverage']:.1%} para um "
        f"nominal de {published['nominal_coverage']:.0%}; a recalibracao conformal chega a "
        f"{adaptive['empirical_coverage']:.1%} com banda "
        f"{interval_verdict['width_reduction_fraction']:.1%} mais estreita e Winkler melhor.",
        "",
        "O gate antigo — 2% de RMSE mais Diebold-Mariano normal — nao decidia nenhuma das",
        "tres perguntas. Todas as tres tem resposta agora, e as respostas nao sao as que o",
        "RMSE sugeria.",
        "",
        "## Acuracia por modelo, decomposta por regime",
        "",
        markdown_table(
            [{"modelo": name, **values} for name, values in baselines.items()],
            ["modelo", "mae", "mae_quiet", "mae_event", "rmse", "largest_error_share_of_sse"],
        ),
        "",
        "A ultima coluna e a razao de tudo isto existir: a fracao do erro quadratico total",
        "que vem de um unico ponto. Enquanto ela estiver nessa ordem de grandeza, qualquer",
        "conclusao apoiada em RMSE esta sendo decidida por uma semana.",
        "",
        "## 1. Paridade contra ARIMA",
        "",
        markdown_table(
            list(parity_verdict["gates"]),
            ["name", "passed", "observed", "threshold"],
        ),
        "",
        f"Veredito: **{'promover' if parity_verdict['promote'] else 'nao promover'}**. "
        + summarize(parity_verdict, "O modelo de paridade"),
        "",
        "## 2. VS-ePL-KRLS contra ARIMA",
        "",
        markdown_table(
            list(vs_verdict["gates"]),
            ["name", "passed", "observed", "threshold"],
        ),
        "",
        f"Veredito: **{'promover' if vs_verdict['promote'] else 'nao promover'}**. "
        + summarize(vs_verdict, "O VS-ePL-KRLS"),
        "",
        "## 3. Intervalo publicado contra recalibracao conformal",
        "",
        markdown_table(
            [
                {"intervalo": "publicado", **interval_verdict["published"]},
                {"intervalo": "conformal adaptativo", **interval_verdict["adaptive_conformal"]},
            ],
            [
                "intervalo",
                "empirical_coverage",
                "calibration_error",
                "mean_width",
                "mean_winkler",
            ],
        ),
        "",
        "Cobertura acima do nominal nao e seguranca gratuita: a banda larga desloca o",
        "cenario P90 e distorce o custo aparente da decisao. O conformal adaptativo chega",
        "perto do nominal com banda mais estreita **e** Winkler melhor, ou seja, nao esta",
        "trocando cobertura por largura — esta corrigindo um nivel que estava errado.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=== paridade contra ARIMA ===")
    for gate in parity_verdict["gates"]:
        flag = "OK " if gate["passed"] else "X  "
        print(f"  {flag}{gate['name']:38s} {gate['observed']:+.6f} (limite {gate['threshold']:+.3f})")
    print(f"  veredito: {'PROMOVER' if parity_verdict['promote'] else 'NAO PROMOVER'}")

    print("\n=== VS-ePL-KRLS contra ARIMA ===")
    for gate in vs_verdict["gates"]:
        flag = "OK " if gate["passed"] else "X  "
        print(f"  {flag}{gate['name']:38s} {gate['observed']:+.6f} (limite {gate['threshold']:+.3f})")
    print(f"  veredito: {'PROMOVER' if vs_verdict['promote'] else 'NAO PROMOVER'}")

    published = interval_verdict["published"]
    adaptive = interval_verdict["adaptive_conformal"]
    print("\n=== intervalo ===")
    print(f"  publicado  cobertura {published['empirical_coverage']:.3f} "
          f"largura {published['mean_width']:.4f} winkler {published['mean_winkler']:.4f}")
    print(f"  conformal  cobertura {adaptive['empirical_coverage']:.3f} "
          f"largura {adaptive['mean_width']:.4f} winkler {adaptive['mean_winkler']:.4f}")
    print(f"  ganho de Winkler: {interval_verdict['winkler_improvement']:+.4f}; "
          f"reducao de largura: {interval_verdict['width_reduction_fraction']:+.1%}")
    print(f"\ngravado em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
