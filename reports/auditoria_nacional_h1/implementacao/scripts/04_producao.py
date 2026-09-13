from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RES = ROOT / "results"
REP = ROOT / "reports"


def validate_forecast_payload(forecast: dict, expected_model: str) -> None:
    required = {
        "modelo",
        "ultima_semana_observada",
        "preco_observado_ultima_semana",
        "previsao_pontual",
        "p10",
        "p90",
        "probabilidades",
    }
    missing = sorted(required - set(forecast))
    if missing:
        raise RuntimeError(f"Artefato de previsao incompleto: {missing}")
    if forecast["modelo"] != expected_model:
        raise RuntimeError(
            "Artefato de previsao inconsistente: "
            f"ranking selecionou {expected_model}, mas o payload contem {forecast['modelo']!r}"
        )
    point, p10, p90 = (float(forecast[key]) for key in ("previsao_pontual", "p10", "p90"))
    if not all(math.isfinite(value) for value in (point, p10, p90)):
        raise RuntimeError("Artefato de previsao contem valores nao finitos")
    if not p10 <= point <= p90:
        raise RuntimeError("Previsao pontual deve estar dentro do intervalo P10-P90")
    probs = forecast["probabilidades"]
    expected_probs = {"p_alta", "p_estavel", "p_queda"}
    if set(probs) != expected_probs:
        raise RuntimeError("Probabilidades devem conter alta, estavel e queda")
    values = [float(probs[key]) for key in sorted(expected_probs)]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise RuntimeError("Probabilidades fora do intervalo [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise RuntimeError("Probabilidades nao somam 1")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.5f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    REP.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RES / "semanal_benchmarks.csv")
    h1 = table[table.horizon == 1].sort_values("rmse")
    winner = h1.iloc[0]
    forecast = json.loads((RES / "previsao_proxima_semana.json").read_text(encoding="utf-8"))
    rec = winner["model"]
    validate_forecast_payload(forecast, rec)
    note = (
        "Modelo recomendado para producao com base no RMSE walk-forward de 1 semana. "
        "Se VS-ePL-KRLS nao for o vencedor, ele permanece como candidato evolutivo "
        "porque atualiza a cada observacao e fornece sinais de drift via beta e regras."
    )
    lines = [
        "# Bloco 3 — Modelo final de producao",
        "",
        f"**Recomendado (h=1 semana, criterio RMSE walk-forward): {rec}**",
        "",
        note,
        "",
        "## Ranking h=1",
        "",
        md_table(h1[["model", "rmse", "mae", "smape", "dir_acc", "coverage_p10_p90"]]),
        "",
        "## Previsao da proxima semana (preco medio nacional de revenda, R$/L)",
        "",
        f"- Semana observada: {forecast.get('ultima_semana_observada')}",
        f"- Preco observado: {forecast.get('preco_observado_ultima_semana')}",
        f"- Previsao pontual: {forecast.get('previsao_pontual')}",
        f"- P10: {forecast.get('p10')}",
        f"- P90: {forecast.get('p90')}",
        f"- Prob. alta / estavel / queda: {forecast.get('probabilidades')}",
        "",
        "## Model card",
        "",
        "- Alvo: preco medio nacional de *revenda* do Diesel B S-10 (ANP), nao o preco de um posto.",
        "- Horizonte de producao: 1 semana a frente.",
        "- Frequencia de atualizacao: incremental a cada nova semana da ANP.",
        "- Exogenas: Brent, USD/BRL, diesel internacional (se disponivel), defasagens, medias moveis, volatilidade, proxy de reajuste Petrobras.",
        "- Preco de distribuicao: usado so na reproducao do artigo; serie ANP termina em ago/2020.",
        "- Intervalo P10-P90: quantis conformais dos residuos walk-forward, nao intervalos gaussianos.",
        "- Limitacoes: buraco ANP ago-out/2020; mudancas de politica de precos; o modelo nao antecipa reajuste da Petrobras no mesmo dia em que e anunciado se a semana ainda nao fechou.",
        "- Quando reajustar: alerta Page-Hinkley, PSI alto nas exogenas, ou degradacao do RMSE movel de 12 semanas.",
        "",
        "Este bloco nao declara reproducao do artigo. A reproducao esta no relatorio 01.",
    ]
    (REP / "03_producao.md").write_text("\n".join(lines), encoding="utf-8")
    print((REP / "03_producao.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
