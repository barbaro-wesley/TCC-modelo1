"""Lucratividade, acerto de movimento e acerto de preco, sobre a evidencia existente.

Nao treina, nao seleciona e nao reabre holdout nenhum: le previsoes ja emitidas e
publicadas e as pontua com as tres taxas que um comprador pergunta.

As janelas nao sao intercambiaveis e o relatorio diz qual e qual:

- **holdout nacional**, 104 semanas, ja lido duas vezes — otimismo de reuso
  embutido;
- **desenvolvimento estadual**, 156 semanas, com o holdout do estado ainda
  fechado;
- **prospectivo**, que e o unico que decide daqui em diante e hoje esta em zero.

Uso::

    python scripts/31_s10_performance.py
    python scripts/31_s10_performance.py --monthly-liters 1000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.audit import FORECAST_EVENTS, verify_audit_ledger  # noqa: E402
from vs_epl_krls.performance import performance_report  # noqa: E402
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402

NEWLINE = chr(10)
REPORTS = ROOT / "reports" / "vs_epl_krls"


def weekly_savings(
    frame: pd.DataFrame, column: str, monthly_liters: float
) -> tuple[np.ndarray, int, int]:
    """Economia semanal, disparos e acertos, quebrando nas lacunas da ANP."""

    # ``dict.fromkeys`` remove a duplicata quando o proprio modelo avaliado e a
    # persistencia: selecionar a mesma coluna duas vezes devolveria um DataFrame
    # onde o replay espera uma Series.
    needed = list(dict.fromkeys(["target_date", "actual", "persistence", column]))
    subset = frame[needed].dropna()
    subset = subset.sort_values("target_date").reset_index(drop=True)
    breaks = subset["target_date"].diff() != pd.Timedelta(days=7)
    blocks = [b for _, b in subset.groupby(breaks.cumsum()) if len(b) >= 3]
    series: list[float] = []
    triggers = wins = 0
    for block in blocks:
        replay = simulate_one_week_prebuy(
            block,
            prediction_column=column,
            model_name=column,
            monthly_liters=monthly_liters,
        )
        series.extend(replay.weekly_net_savings_brl)
        triggers += replay.triggered_prebuys
        if replay.trigger_precision is not None:
            wins += round(replay.trigger_precision * replay.triggered_prebuys)
    return np.asarray(series, dtype=float), int(triggers), int(wins)


def national_holdout(monthly_liters: float) -> list:
    frame = pd.read_csv(
        REPORTS / "s10_parity" / "holdout_predictions.csv", parse_dates=["target_date"]
    )
    weekly_litres = monthly_liters * 12.0 / 52.0
    spend = float(weekly_litres * frame["actual"].sum())
    reports = []
    for label, column, bands in (
        ("paridade", "paridade", ("paridade__lower", "paridade__upper")),
        ("ARIMA", "arima", None),
        ("persistencia", "persistence", None),
    ):
        savings, triggers, wins = weekly_savings(frame, column, monthly_liters)
        reports.append(
            performance_report(
                model=label,
                window="holdout nacional, 104 semanas",
                actual=frame["actual"],
                prediction=frame[column],
                origin=frame["persistence"],
                lower=frame[bands[0]] if bands else None,
                upper=frame[bands[1]] if bands else None,
                weekly_savings=savings,
                triggers=triggers,
                winning_triggers=wins,
                baseline_spend_brl=spend,
                notes=(
                    "holdout lido duas vezes; o otimismo de reuso esta embutido",
                ),
            )
        )
    return reports


def state_development(uf: str, monthly_liters: float) -> list:
    path = REPORTS / f"s10_{uf.lower()}" / "development_predictions.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path, parse_dates=["target_date"])
    weekly_litres = monthly_liters * 12.0 / 52.0
    spend = float(weekly_litres * frame["actual"].sum())
    reports = []
    for label, column in (
        (f"{uf} nacional+spread", "nacional+spread"),
        (f"{uf} direto", "rs_direto"),
        (f"{uf} persistencia", "persistencia"),
    ):
        if column not in frame.columns:
            continue
        savings, triggers, wins = weekly_savings(frame, column, monthly_liters)
        reports.append(
            performance_report(
                model=label,
                window=f"desenvolvimento {uf}, 156 semanas",
                actual=frame["actual"],
                prediction=frame[column],
                origin=frame["persistence"],
                weekly_savings=savings,
                triggers=triggers,
                winning_triggers=wins,
                baseline_spend_brl=spend,
                notes=("holdout estadual nunca aberto",),
            )
        )
    return reports


def prospective() -> dict[str, object]:
    """O que os ledgers ja permitem afirmar — hoje, quase nada, e isso importa."""

    summary: dict[str, object] = {}
    for name, path in (
        ("paridade", REPORTS / "s10_parity" / "parity_ledger.jsonl"),
        ("regional_rs", REPORTS / "s10_rs" / "rs_ledger.jsonl"),
    ):
        if not path.is_file():
            continue
        records = verify_audit_ledger(path)
        settled = [r for r in records if r["event"] == "realized"]
        issued = [r for r in records if r["event"] in FORECAST_EVENTS]
        summary[name] = {
            "forecasts": len(issued),
            "settled": len(settled),
            "movement_hit_rate": (
                float(
                    np.mean(
                        [
                            np.sign(
                                float(r["payload"]["point"])
                                - float(r["payload"]["observed_price"])
                                + float(r["payload"]["persistence_absolute_error"])
                            )
                            != 0
                            for r in settled
                        ]
                    )
                )
                if settled
                else None
            ),
            "mae": (
                float(np.mean([float(r["payload"]["absolute_error"]) for r in settled]))
                if settled
                else None
            ),
        }
    return summary


def render(report) -> list[str]:
    lines = [f"### {report.model} — {report.window}", ""]
    profit = report.profitability
    if profit is not None:
        lines += [
            "**Lucratividade**",
            "",
            f"- disparos: **{profit.triggers}** em {profit.weeks} semanas, "
            f"{profit.winning_triggers} com lucro",
            "- taxa de acerto do disparo: "
            + (f"**{profit.win_rate:.1%}**" if profit.win_rate is not None else "—"),
            "- fator de lucro: "
            + (
                f"**{profit.profit_factor:.2f}**"
                if profit.profit_factor is not None and np.isfinite(profit.profit_factor)
                else ("sem perdas registradas" if profit.gross_loss_brl == 0 else "—")
            ),
            "- expectativa por disparo: "
            + (
                f"**R$ {profit.expectancy_brl:,.2f}**"
                if profit.expectancy_brl is not None
                else "—"
            ),
            f"- economia liquida: R$ {profit.net_savings_brl:,.2f} "
            f"(ganho bruto R$ {profit.gross_profit_brl:,.2f}, perda bruta "
            f"R$ {profit.gross_loss_brl:,.2f})",
            "- retorno sobre o gasto: "
            + (
                f"**{profit.return_on_spend:.4%}**"
                if profit.return_on_spend is not None
                else "—"
            ),
        ]
        if profit.largest_event_share is not None:
            lines.append(
                f"- concentracao no maior evento: {profit.largest_event_share:.1%}"
            )
        lines.append("")
    move = report.movement
    if move is not None:
        lines += [
            "**Acerto de movimento**",
            "",
            "- semanas que se moveram: "
            + (f"**{move.hit_rate_moved:.1%}**" if move.hit_rate_moved is not None else "—")
            + f" ({move.n_moved} semanas)",
            f"- semanas de evento (>{move.threshold_brl:.2f} R$/L): "
            + (
                f"**{move.hit_rate_events:.1%}**"
                if move.hit_rate_events is not None
                else "—"
            )
            + f" ({move.n_events} semanas)",
            "",
        ]
    price = report.price
    if price is not None:
        ladder = "  ".join(
            f"±R$ {float(k):.2f}: **{v:.1%}**" for k, v in price.within.items()
        )
        lines += [
            "**Acerto de preco**",
            "",
            f"- {ladder}",
            f"- erro mediano: R$ {price.median_absolute_error:.4f}/L; "
            f"medio: R$ {price.mean_absolute_error:.4f}/L",
        ]
        if price.interval_coverage is not None:
            lines.append(
                f"- cobertura do intervalo: {price.interval_coverage:.1%} "
                f"(nominal {price.interval_nominal:.0%})"
            )
        lines.append("")
    for note in report.notes:
        lines.append(f"> {note}")
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monthly-liters", type=float, default=200_000.0)
    parser.add_argument("--uf", action="append", default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=REPORTS / "s10_performance"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = national_holdout(args.monthly_liters)
    for uf in args.uf or ["RS"]:
        reports.extend(state_development(str(uf).strip().upper(), args.monthly_liters))

    forward = prospective()

    table = []
    for report in reports:
        row: dict[str, object] = {"modelo": report.model, "janela": report.window}
        if report.profitability:
            row["disparos"] = report.profitability.triggers
            row["acerto_disparo"] = report.profitability.win_rate
            row["fator_lucro"] = report.profitability.profit_factor
            row["retorno_sobre_gasto"] = report.profitability.return_on_spend
        if report.movement:
            row["acerto_movimento"] = report.movement.hit_rate_moved
            row["acerto_evento"] = report.movement.hit_rate_events
        if report.price:
            row["preco_ate_2c"] = report.price.within["0.02"]
            row["preco_ate_5c"] = report.price.within["0.05"]
            row["cobertura"] = report.price.interval_coverage
        table.append(row)
    frame = pd.DataFrame(table)

    print(f"volume de referencia: {args.monthly_liters:,.0f} L/mes")
    print()
    columns = [
        c
        for c in (
            "modelo", "disparos", "acerto_disparo", "fator_lucro",
            "retorno_sobre_gasto", "acerto_movimento", "acerto_evento",
            "preco_ate_2c", "preco_ate_5c",
        )
        if c in frame.columns
    ]
    print(frame[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("prospectivo (o unico que decide daqui em diante):")
    for name, values in forward.items():
        print(f"  {name}: {values['settled']} semana(s) liquidada(s) de "
              f"{values['forecasts']} previsao(oes)")

    payload = {
        "monthly_liters": args.monthly_liters,
        "definitions": {
            "lucratividade": (
                "sobre os disparos da politica: taxa de acerto, fator de lucro, "
                "expectativa por disparo e retorno sobre o gasto com combustivel"
            ),
            "acerto_de_movimento": (
                "fracao das semanas em que a direcao prevista bateu com a realizada, "
                "excluindo semanas paradas; reportado tambem so nas semanas de evento"
            ),
            "acerto_de_preco": (
                "fracao das semanas com erro dentro de 1, 2, 5 e 10 centavos por litro, "
                "mais a cobertura do intervalo P10-P90"
            ),
        },
        "reports": [report.as_dict() for report in reports],
        "prospective": forward,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    frame.to_csv(args.output_dir / "summary.csv", index=False)

    document = [
        "# Lucratividade, acerto de movimento e acerto de preco",
        "",
        f"Volume de referencia: **{args.monthly_liters:,.0f} L/mes**. Todas as taxas saem de",
        "previsoes ja emitidas; nada aqui treina ou seleciona modelo.",
        "",
        "## As definicoes, antes dos numeros",
        "",
        "**Lucratividade** e medida sobre os *disparos* da politica de antecipacao, na",
        "convencao de mercado: quantos deram lucro, quanto o ganho total supera a perda",
        "total (fator de lucro), quanto rende um disparo medio (expectativa) e quanto isso",
        "representa sobre o gasto com combustivel. Taxa de acerto alta com expectativa baixa",
        "e a armadilha classica; por isso os quatro numeros andam juntos.",
        "",
        "**Acerto de movimento** exclui as semanas paradas. Acertar que nada aconteceria nao",
        "e previsao, e incluir isso infla o numero — dois tercos das semanas desta serie sao",
        "paradas.",
        "",
        "**Acerto de preco** nao tem resposta unica sem escolher uma tolerancia, entao vem a",
        "escada inteira. Escolha a tolerancia que corresponde a sua decisao, e diga qual e.",
        "",
        "## Aviso sobre as janelas",
        "",
        "As janelas **nao sao comparaveis entre si**. O holdout nacional foi lido duas vezes",
        "e carrega otimismo de reuso. O desenvolvimento estadual e desenvolvimento, nao teste",
        "cego. E o prospectivo — o unico que decide de agora em diante — esta em zero semanas",
        "liquidadas.",
        "",
    ]
    for report in reports:
        document.extend(render(report))
    document += [
        "## Prospectivo",
        "",
    ]
    for name, values in forward.items():
        document.append(
            f"- **{name}**: {values['settled']} semana(s) liquidada(s) de "
            f"{values['forecasts']} previsao(oes) registrada(s)."
        )
    document += [
        "",
        "Enquanto essa contagem for zero, **nenhuma taxa acima e evidencia prospectiva** —",
        "sao todas retrospectivas, e o holdout nacional ja foi gasto.",
        "",
    ]
    (args.output_dir / "report.md").write_text(
        NEWLINE.join(document) + NEWLINE, encoding="utf-8"
    )
    print(f"{NEWLINE}gravado em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
