from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import summarize  # noqa: E402
from vsepl_krls.model import VSePLKRLS, VSePLKRLSConfig  # noqa: E402
from vsepl_krls.paper import (  # noqa: E402
    EPL_KRLS_S10_H1,
    S10_MONTHLY_H1,
    S10_MONTHLY_H6,
    S10_MONTHLY_H12,
    make_supervised,
    monthly_params,
    reproduction_verdict,
    scale_inputs_train_only,
)

PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"


def run_one(X, y, n_train, cfg: VSePLKRLSConfig, seed_update: bool = True):
    Xtr, Xte, _ = scale_inputs_train_only(X[:n_train], X[n_train:])
    ytr, yte = y[:n_train], y[n_train:]
    model = VSePLKRLS(cfg)
    for i in range(len(ytr)):
        model.update(Xtr[i], float(ytr[i]))
    preds = []
    for i in range(len(yte)):
        yhat = model.predict_one(Xte[i])
        preds.append(yhat)
        if seed_update:
            model.update(Xte[i], float(yte[i]))
    preds = np.asarray(preds, dtype=float)
    metrics = summarize(yte, preds)
    metrics["n_rules"] = float(model.n_rules)
    metrics["beta_final"] = float(model.beta)
    metrics["ndei_full"] = float(metrics["rmse"] / 0.453)
    return metrics, preds, yte, model


def experiment(horizon: int, convention: str, vs: bool):
    df = pd.read_csv(PROC / "mensal_s10_artigo.csv", parse_dates=["data"])
    X, y, idx = make_supervised(df["revenda"].to_numpy(), df["distribuicao"].to_numpy(), horizon)
    extra = monthly_params(horizon) if vs else dict(gamma_bar=0.006, alpha_vs1=0.88, alpha_vs2=0.74)
    n_train = {1: 72, 6: 66, 12: 60}[horizon]
    cfg = VSePLKRLSConfig(
        threshold_convention=convention,
        use_variable_step=vs,
        **extra,
    )
    metrics, preds, yte, model = run_one(X, y, n_train, cfg)
    return {
        "horizon": horizon,
        "convention": convention,
        "variable_step": vs,
        "n_pairs": int(len(y)),
        "n_train": n_train,
        "n_test": int(len(yte)),
        "metrics": metrics,
        "preds": preds.tolist(),
        "y_test": yte.tolist(),
        "n_rules_path": model.history_n_rules,
        "beta_path": model.history_beta,
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "mensal_s10_artigo.csv", parse_dates=["data"])
    print(f"Janela artigo: {df['data'].min().date()} -> {df['data'].max().date()} ({len(df)} obs)")

    rows = []
    artifacts = {}
    for horizon, target in [(1, S10_MONTHLY_H1), (6, S10_MONTHLY_H6), (12, S10_MONTHLY_H12)]:
        for convention in ("tabela", "texto"):
            for vs in (True, False):
                name = f"h{horizon}_{convention}_{'vs' if vs else 'epl'}"
                print(f"Rodando {name}...")
                art = experiment(horizon, convention, vs)
                tgt = target if vs else (EPL_KRLS_S10_H1 if horizon == 1 else target)
                # NDEI do artigo usa o desvio da serie (Tabela 1), nao so o teste
                art["metrics"]["ndei"] = art["metrics"].get("ndei_full", art["metrics"]["ndei"])
                verdict = reproduction_verdict(art["metrics"], tgt) if (vs or horizon == 1) else "ablacao"
                art["target"] = {
                    "rmse": tgt.rmse, "mae": tgt.mae, "ndei": tgt.ndei, "n_rules": tgt.n_rules
                }
                art["verdict"] = verdict
                artifacts[name] = art
                m = art["metrics"]
                rows.append(
                    {
                        "experimento": name,
                        "horizonte": horizon,
                        "convencao": convention,
                        "VS": vs,
                        "rmse": m["rmse"],
                        "mae": m["mae"],
                        "ndei": m["ndei"],
                        "n_regras": m["n_rules"],
                        "veredito": verdict,
                        "rmse_artigo": tgt.rmse,
                        "mae_artigo": tgt.mae,
                        "ndei_artigo": tgt.ndei,
                    }
                )
                print(
                    f"  RMSE={m['rmse']:.5f} MAE={m['mae']:.5f} NDEI={m['ndei']:.5f} "
                    f"regras={int(m['n_rules'])} -> {verdict}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(RES / "reproducao_mensal.csv", index=False)
    (RES / "reproducao_mensal.json").write_text(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("n_rules_path", "beta_path")}
                    for k, v in artifacts.items()}, indent=2),
        encoding="utf-8",
    )

    best_h1 = table[(table.horizonte == 1) & (table.VS)].sort_values("rmse").iloc[0]
    key = best_h1["experimento"]
    art = artifacts[key]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(art["y_test"], marker="o", label="Real")
    ax.plot(art["preds"], marker="x", label="VS-ePL-KRLS")
    ax.set_title(f"Reproducao S10 mensal h=1 ({key})\nveredito: {art['verdict']}")
    ax.set_xlabel("Mes de teste")
    ax.set_ylabel("R$/L")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "reproducao_h1.png", dpi=140)
    plt.close(fig)

    lines = [
        "# Bloco 1 — Reproducao do artigo",
        "",
        "Janela historica: dezembro/2012 a maio/2020, Diesel S-10 nacional, previsao mensal.",
        "Entrada: `[preco_distribuicao(t), preco_revenda(t)]`. Alvo: `preco_revenda(t+h)`.",
        "Normalizacao min-max apenas no treino. Teste: prever antes de atualizar.",
        "",
        "## Criterio (fixado antes de rodar)",
        "REPRODUZIDO se RMSE, MAE e NDEI ficarem a ±10% dos valores publicados e o numero de regras coincidir.",
        "",
        "## Resultados",
        "",
        table.to_string(index=False),
        "",
        f"Melhor configuracao em h=1: **{key}** com veredito **{art['verdict']}**.",
        "",
        "Nenhuma afirmacao de reproducao e feita fora desta tabela. Os numeros deste bloco",
        "nao se transferem automaticamente para a frequencia semanal (bloco 2).",
    ]
    try:
        md = "\n".join(lines)
    except Exception:
        md = "\n".join(lines[:7] + [table.to_string(index=False)] + lines[-5:])
    (REP / "01_reproducao.md").write_text(md, encoding="utf-8")
    print("\nRelatorio:", REP / "01_reproducao.md")


if __name__ == "__main__":
    main()
