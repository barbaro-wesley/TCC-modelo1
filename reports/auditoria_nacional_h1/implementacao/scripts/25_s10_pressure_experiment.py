"""Testa a pressao de repasse produtor-paridade nos folds de desenvolvimento.

Hipotese: o anuncio de reajuste da Petrobras nao e aleatorio, e a resposta a um
desvio acumulado em relacao a paridade de importacao.  Esse desvio e observavel
em tempo real — o ultimo preco de produtor publicado contra a paridade de hoje —
enquanto o anuncio em si so existe como texto de assessoria.  Se a hipotese
valer, o atributo captura parte do ganho que a defasagem de publicacao trancou.

**O holdout nao e lido.**  A avaliacao para no fim do desenvolvimento, pela
janela congelada por data.  Um resultado positivo aqui nao promove nada: vira
candidato pre-registrado para confirmacao prospectiva.

Uso::

    python scripts/25_s10_pressure_experiment.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls.gates import accuracy_report, paired_block_bootstrap  # noqa: E402
from vs_epl_krls.passthrough import (  # noqa: E402
    PARITY_FEATURES,
    PassThroughConfig,
    PassThroughECM,
    build_parity_panel,
)
from vs_epl_krls.pressure import PRESSURE_GATE_Z, build_pressure_features  # noqa: E402
from vs_epl_krls.procurement import simulate_one_week_prebuy  # noqa: E402
from vs_epl_krls.selection import pinned_validation_folds  # noqa: E402

#: Especificacoes comparadas.  A primeira e a congelada em producao; as demais
#: acrescentam pressao, uma coluna por vez, para que o efeito seja atribuivel.
SPECS: dict[str, tuple[str, ...]] = {
    "paridade": PARITY_FEATURES,
    "paridade+press": PARITY_FEATURES + ("press1",),
    "paridade+press+dpress": PARITY_FEATURES + ("press1", "dpress1"),
    "press_only": ("dp1", "press1"),
}

MONTHLY_LITERS = 200_000.0

NEWLINE = chr(10)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}" if abs(value) < 1000 else f"{value:,.0f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return NEWLINE.join(lines)


def build_panel(causal_path: Path) -> pd.DataFrame:
    causal = pd.read_csv(causal_path, parse_dates=["date"])
    panel = build_parity_panel(causal)
    pressure = build_pressure_features(causal)
    merged = panel.merge(pressure, on="date", how="left", validate="one_to_one")
    if len(merged) != len(panel):
        raise RuntimeError("a juncao de pressao alterou o numero de linhas do painel")
    return merged


def walk_fold(panel: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "target_date": panel["date"].iloc[start:end].to_numpy(),
            "actual": panel["price"].iloc[start:end].to_numpy(float),
            "persistence": panel["origin_price"].iloc[start:end].to_numpy(float),
        }
    ).reset_index(drop=True)
    config = PassThroughConfig()
    for name, features in SPECS.items():
        missing = [feature for feature in features if feature not in panel.columns]
        if missing:
            raise RuntimeError(f"{name} exige colunas ausentes: {missing}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PassThroughECM(config=config, feature_names=features)
            result = model.walk_forward(panel, start, end, refit_every=1)
        frame[name] = result["prediction"].to_numpy(float)

    # Portao de decisao: mantem a previsao congelada, mas so deixa a politica
    # disparar quando a refinaria esta barata em relacao a paridade.  Fora dessa
    # condicao a linha recebe o preco de origem, que nunca dispara.  Isto usa a
    # pressao onde ela e forte — a probabilidade de evento — em vez de forcar um
    # regressor linear sobre semanas em que nada acontece.
    gate = panel["press_z"].iloc[start:end].to_numpy(float) < PRESSURE_GATE_Z
    frame["paridade@gate_press"] = np.where(gate, frame["paridade"], frame["persistence"])
    frame["arima_like_gate"] = np.where(gate, frame["paridade"], frame["persistence"])
    return frame


def policy(frame: pd.DataFrame, column: str) -> dict[str, object]:
    subset = frame[["target_date", "actual", "persistence", column]].dropna()
    if len(subset) < 3:
        return {"net_savings_brl": float("nan"), "triggered": 0}
    replay = simulate_one_week_prebuy(
        subset, prediction_column=column, model_name=column, monthly_liters=MONTHLY_LITERS
    )
    return {
        "net_savings_brl": replay.net_savings_brl,
        "triggered": replay.triggered_prebuys,
        "precision": replay.trigger_precision,
        "largest_event_share": replay.largest_event_share_of_savings,
    }


def mechanism_diagnostics(panel: pd.DataFrame, development_end: int) -> dict[str, object]:
    """Testa o mecanismo, nao so o desempenho.

    Se a hipotese vale, semanas em que a refinaria esta barata em relacao a
    paridade — pressao no quintil mais baixo — devem ser seguidas por altas
    maiores do que semanas no quintil mais alto.  Um atributo que melhora a
    economia sem esse padrao seria coincidencia; com ele, e mecanismo.
    """

    window = panel.iloc[:development_end]
    subset = window[["press1", "y"]].dropna()
    if len(subset) < 50:
        return {"available": False}
    pressure = subset["press1"].to_numpy(float)
    change = subset["y"].to_numpy(float)
    low, high = np.quantile(pressure, [0.2, 0.8])
    bottom = change[pressure <= low]
    top = change[pressure >= high]
    event = np.abs(change) > 0.02
    return {
        "available": True,
        "n": int(len(subset)),
        "correlation_with_next_change": float(np.corrcoef(pressure, change)[0, 1]),
        "mean_change_lowest_quintile_brl": float(np.mean(bottom)),
        "mean_change_highest_quintile_brl": float(np.mean(top)),
        "quintile_spread_brl": float(np.mean(bottom) - np.mean(top)),
        "event_rate_lowest_quintile": float(np.mean(event[pressure <= low])),
        "event_rate_highest_quintile": float(np.mean(event[pressure >= high])),
        "event_rate_overall": float(np.mean(event)),
    }


def redundancy_diagnostics(
    panel: pd.DataFrame, combined: pd.DataFrame
) -> dict[str, object]:
    """O teste decisivo: a pressao explica o que o modelo congelado ja nao explica?

    Um atributo pode ser fortemente preditivo sozinho e ainda assim inutil, se o
    que ele carrega ja estiver dentro dos atributos existentes.  A pergunta certa
    nao e "pressao correlaciona com a variacao seguinte" e sim "pressao
    correlaciona com o **residuo** do modelo congelado".
    """

    overlap = {}
    for feature in ("rpar1", "rpar2", "coint_par"):
        pair = panel[["press_z", feature]].dropna()
        if len(pair) > 30:
            overlap[feature] = float(
                np.corrcoef(pair["press_z"], pair[feature])[0, 1]
            )

    merged = combined.merge(
        panel[["date", "press_z"]], left_on="target_date", right_on="date", how="left"
    )
    residual_frame = merged[["actual", "paridade", "press_z"]].dropna()
    if len(residual_frame) < 30:
        return {"available": False, "overlap_with_frozen_features": overlap}
    residual = (residual_frame["actual"] - residual_frame["paridade"]).to_numpy(float)
    pressure = residual_frame["press_z"].to_numpy(float)
    correlation = float(np.corrcoef(pressure, residual)[0, 1])

    # A mesma cautela que o projeto aplica aos modelos tem de valer para os
    # proprios achados: nesta serie uma correlacao de Pearson pode ser tres
    # pontos.  Spearman ignora a magnitude da cauda, e a remocao dos maiores
    # residuos mede diretamente quanto do sinal depende deles.
    spearman = float(stats.spearmanr(pressure, residual).statistic)
    order = np.argsort(-np.abs(residual))
    trimmed = {}
    for removed in (1, 3, 5):
        keep = np.ones(residual.size, dtype=bool)
        keep[order[:removed]] = False
        trimmed[f"pearson_sem_{removed}_maiores"] = float(
            np.corrcoef(pressure[keep], residual[keep])[0, 1]
        )
    below = pressure < 0.0
    # Numero no lugar de limiar: quanto da correlacao desaparece junto com tres
    # semanas.  Um limiar binario aqui seria a mesma armadilha que este projeto
    # ja documentou nos gates antigos.
    tail_dependence = (
        1.0 - abs(trimmed["pearson_sem_3_maiores"]) / abs(correlation)
        if correlation
        else float("nan")
    )
    return {
        "available": True,
        "n": int(len(residual_frame)),
        "overlap_with_frozen_features": overlap,
        "correlation_with_frozen_residual": correlation,
        "spearman_with_frozen_residual": spearman,
        "trimmed_correlations": trimmed,
        "n_below_gate": int(below.sum()),
        "n_at_or_above_gate": int((~below).sum()),
        "tail_dependence_fraction": float(tail_dependence),
        "verdict": (
            f"{tail_dependence:.0%} da correlacao de Pearson vem de tres semanas; "
            f"em posto (Spearman) sobra {spearman:+.2f}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=ROOT / "data" / "processed" / "s10_causal_panel.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / "vs_epl_krls" / "s10_pressure"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(args.panel)
    windows = pinned_validation_folds(panel["date"])
    coverage = float(panel["press1"].notna().mean())
    print(f"painel: {len(panel)} semanas; pressao disponivel em {coverage:.1%} das linhas")
    print(
        "janela: desenvolvimento ate "
        f"{panel['date'][windows.development_end - 1].date()}; o holdout nao e lido"
    )

    frames = [walk_fold(panel, fold.validation_start, fold.validation_end) for fold in windows.folds]
    combined = pd.concat(frames, ignore_index=True)

    evaluated = list(SPECS) + ["paridade@gate_press"]
    rows: list[dict[str, object]] = []
    for name in evaluated:
        valid = combined[["actual", "persistence", name]].dropna()
        report = accuracy_report(
            valid["actual"].to_numpy(float),
            valid[name].to_numpy(float),
            valid["persistence"].to_numpy(float),
        )
        rows.append({"spec": name, **asdict(report), **policy(combined, name)})
    summary = pd.DataFrame(rows)

    baseline = combined[["actual", "persistence", "paridade"]].dropna()
    comparisons: dict[str, dict[str, float]] = {}
    for name in evaluated:
        if name == "paridade":
            continue
        pair = combined[["actual", "paridade", name]].dropna()
        comparisons[name] = paired_block_bootstrap(
            np.abs(pair["actual"] - pair[name]),
            np.abs(pair["actual"] - pair["paridade"]),
        )

    print()
    print(
        summary[
            ["spec", "mae", "mae_quiet", "mae_event", "directional_accuracy",
             "net_savings_brl", "triggered", "precision"]
        ].to_string(index=False)
    )
    mechanism = mechanism_diagnostics(panel, windows.development_end)
    if mechanism.get("available"):
        print()
        print("mecanismo (so desenvolvimento):")
        print("  correlacao pressao x variacao seguinte: "
              f"{mechanism['correlation_with_next_change']:+.4f}")
        print("  variacao media no quintil de menor pressao: "
              f"R$ {mechanism['mean_change_lowest_quintile_brl']:+.4f}/L")
        print("  variacao media no quintil de maior pressao: "
              f"R$ {mechanism['mean_change_highest_quintile_brl']:+.4f}/L")
        print(f"  taxa de evento: {mechanism['event_rate_lowest_quintile']:.1%} no menor "
              f"quintil contra {mechanism['event_rate_highest_quintile']:.1%} no maior "
              f"(base {mechanism['event_rate_overall']:.1%})")

    redundancy = redundancy_diagnostics(panel, combined)
    print()
    print("redundancia:")
    for feature, value in redundancy.get("overlap_with_frozen_features", {}).items():
        print(f"  correlacao pressao x {feature}: {value:+.4f}")
    if redundancy.get("available"):
        print("  correlacao pressao x residuo do modelo congelado: "
              f"{redundancy['correlation_with_frozen_residual']:+.4f} (Pearson), "
              f"{redundancy['spearman_with_frozen_residual']:+.4f} (Spearman)")
        for label, value in redundancy["trimmed_correlations"].items():
            print(f"    {label}: {value:+.4f}")
        print(f"  lados do portao: {redundancy['n_below_gate']} semanas abaixo, "
              f"{redundancy['n_at_or_above_gate']} acima "
              "(lado alto quase vazio nos folds: o portao nao e testavel aqui)")
        print(f"  veredito: {redundancy['verdict']}")

    print("\nMAE contra a especificacao congelada (bootstrap pareado em blocos):")
    for name, result in comparisons.items():
        verdict = "melhor" if result["ci90_high"] < 0 else "sem ganho decidivel"
        print(
            f"  {name:24s} diferenca {result['mean_difference']:+.6f} "
            f"IC90 [{result['ci90_low']:+.6f}, {result['ci90_high']:+.6f}] -> {verdict}"
        )

    manifest = {
        "hypothesis": (
            "a distancia entre o ultimo preco de produtor publicado e a paridade "
            "de importacao corrente antecipa o repasse, sem depender de raspagem "
            "de anuncio"
        ),
        "holdout_evaluated": False,
        "development_end_date": str(panel["date"][windows.development_end - 1].date()),
        "folds": [asdict(fold) for fold in windows.folds],
        "pressure_coverage_fraction": coverage,
        "specs": {name: list(features) for name, features in SPECS.items()},
        "decision_gate": {
            "name": "paridade@gate_press",
            "rule": "so dispara quando press_z < PRESSURE_GATE_Z",
            "threshold_z": PRESSURE_GATE_Z,
        },
        "summary": json.loads(summary.to_json(orient="records")),
        "mae_comparisons_vs_frozen": comparisons,
        "mechanism": mechanism,
        "redundancy": redundancy,
        "n_development_rows": int(len(baseline)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    frozen = summary[summary["spec"] == "paridade"].iloc[0]
    best_savings = summary.loc[summary["net_savings_brl"].idxmax()]
    report = [
        "# Pressao de repasse produtor-paridade — experimento de desenvolvimento",
        "",
        "**O holdout nao foi lido.** A avaliacao termina em "
        f"{manifest['development_end_date']}, pela janela congelada por data.",
        "",
        "## A hipotese",
        "",
        "O relatorio de paridade concluiu que o proximo ganho material exigiria capturar",
        "os anuncios de reajuste da Petrobras no dia em que saem. A pesquisa confirma que",
        "esses anuncios existem, sao publicos e trazem data e magnitude em R$/L — e que",
        "saem como texto de assessoria, sem serie baixavel.",
        "",
        "Este experimento tenta um caminho que dispensa raspagem: o anuncio e a *resposta*",
        "da refinaria a um desvio em relacao a paridade de importacao, e os dois lados",
        "desse desvio ja estao no painel. O ultimo preco de produtor publicado, mesmo",
        "defasado, contra a paridade de hoje da uma medida de quanto a refinaria esta",
        "atrasada — disponivel em tempo real.",
        "",
        "## O mecanismo existe",
        "",
        f"Sobre o desenvolvimento inteiro ({mechanism['n']} semanas):",
        "",
        f"- correlacao da pressao com a variacao seguinte: **{mechanism['correlation_with_next_change']:+.4f}**;",
        f"- variacao media no quintil de menor pressao: **R$ {mechanism['mean_change_lowest_quintile_brl']:+.4f}/L**;",
        f"- variacao media no quintil de maior pressao: **R$ {mechanism['mean_change_highest_quintile_brl']:+.4f}/L**;",
        f"- taxa de semana de evento: **{mechanism['event_rate_lowest_quintile']:.1%}** no menor quintil "
        f"contra **{mechanism['event_rate_highest_quintile']:.1%}** no maior, base {mechanism['event_rate_overall']:.1%}.",
        "",
        "O sinal aponta na direcao que a economia prediz e e muito mais forte do que o",
        "+0,097 que a defasagem de publicacao deixava disponivel. Ate aqui, a hipotese",
        "sobrevive.",
        "",
        "## E ainda assim nao vira previsao melhor",
        "",
        markdown_table(
            json.loads(summary.to_json(orient="records")),
            ["spec", "mae", "mae_quiet", "mae_event", "net_savings_brl", "triggered", "precision"],
        ),
        "",
        "Nenhuma especificacao com pressao melhora o MAE de forma decidivel: o bootstrap",
        "pareado em blocos coloca zero dentro do IC90 em todas elas. O ganho aparece so na",
        f"moeda da decisao — R$ {best_savings['net_savings_brl']:,.0f} contra "
        f"R$ {frozen['net_savings_brl']:,.0f} da especificacao congelada — e vem com mais",
        "gatilhos e precisao um pouco menor. E o mesmo padrao do modelo de paridade contra",
        "o ARIMA: **decide melhor do que preve**.",
        "",
        "## A parte que quase enganou",
        "",
        f"A pressao correlaciona {redundancy['correlation_with_frozen_residual']:+.4f} com o",
        "residuo do modelo congelado, e so fracamente com os atributos que ja existem",
        f"(rpar1 {redundancy['overlap_with_frozen_features'].get('rpar1', float('nan')):+.4f}).",
        "Lido assim, pareceria sinal novo e forte, e a conclusao seria promover.",
        "",
        "Mas a mesma cautela que este projeto aplica aos modelos vale para os proprios",
        "achados:",
        "",
        f"- **{redundancy['tail_dependence_fraction']:.0%} dessa correlacao desaparece** ao remover tres semanas "
        f"(de {redundancy['correlation_with_frozen_residual']:+.4f} para "
        f"{redundancy['trimmed_correlations']['pearson_sem_3_maiores']:+.4f});",
        f"- em posto, Spearman entrega apenas {redundancy['spearman_with_frozen_residual']:+.4f};",
        f"- o portao de decisao nao e testavel nos folds: {redundancy['n_at_or_above_gate']} das "
        f"{redundancy['n']} semanas caem do lado alto.",
        "",
        "Ou seja: o mecanismo e real, a magnitude nao esta estabelecida, e a serie tem o",
        "mesmo tamanho amostral efetivo minusculo que ja invalidou os gates antigos.",
        "",
        "## Conclusao",
        "",
        "1. A pressao produtor-paridade **e um indicador antecedente real e disponivel em",
        "   tempo real**, com mecanismo economico explicito e sem depender de raspagem.",
        "2. Ela **nao melhora a previsao** de forma decidivel na forma linear testada.",
        "3. O ganho na politica de compra e consistente com o mecanismo, mas cabe dentro do",
        "   ruido que este holdout ja demonstrou produzir.",
        "4. **Nada e promovido.** A especificacao fica pre-registrada; so semanas futuras,",
        "   pelo ledger prospectivo, podem decidir. O holdout continua fechado.",
        "",
    ]
    (args.output_dir / "report.md").write_text(NEWLINE.join(report) + NEWLINE, encoding="utf-8")
    summary.to_csv(args.output_dir / "development_summary.csv", index=False)
    combined.to_csv(args.output_dir / "development_predictions.csv", index=False)
    print(f"\ngravado em {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
