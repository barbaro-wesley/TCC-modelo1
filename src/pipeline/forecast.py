"""Previsao de producao: reajusta cada candidato em toda a serie e publica o vencedor.

O vencedor e o menor RMSE walk-forward em h=1 (results/semanal_benchmarks.csv),
que e o mesmo criterio ja usado no relatorio 03. Antes deste modulo o arquivo
previsao_proxima_semana.json trazia sempre o VS-ePL-KRLS, mesmo quando o ranking
apontava outro modelo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from benchmarks.classical import arima_forecast, arimax_forecast, ma_predict, naive_predict
from benchmarks.gbm import fit_lgbm, fit_xgb
from data.panel import ARIMAX_COLS, latest_features, load_features, load_panel, scale_frozen
from eval.intervals import direction_probs
from vsepl_krls.model import VSePLKRLS, VSePLKRLSConfig

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"

N_MIN = 80  # mesmo n_min do walk-forward em scripts/03_semanal.py
JANELA_RESIDUOS = 80

# LSTM fica de fora: e caro, perde por ordens de grandeza no walk-forward e
# exige subprocesso com torch. Se vencer o ranking, cai para o proximo.
MODELOS_PRODUCAO = ("ARIMA", "ARIMAX", "naive", "media_movel", "LightGBM", "XGBoost", "VS-ePL-KRLS")


def _serie_precos() -> np.ndarray:
    full = load_features()
    return full["revenda"].astype(float).dropna().to_numpy()


def _prever_vsepl(panel: pd.DataFrame, feat_cols: list, x_last: np.ndarray) -> float:
    X = panel[feat_cols].to_numpy(float)
    y = panel["y"].to_numpy(float)
    Xs, lo, span = scale_frozen(X, N_MIN)
    model = VSePLKRLS(
        VSePLKRLSConfig(d_max=15, alpha=0.01, beta0=0.18, max_rules=8, threshold_convention="texto")
    )
    for t in range(len(y)):
        model.update(Xs[t], float(y[t]))
    if not model.n_rules:
        return float("nan")
    xt = np.clip((x_last - lo) / span, -0.25, 1.25)
    return float(model.predict_one(xt))


def _prever_arimax(panel: pd.DataFrame) -> float:
    precos = panel["y_prev"].to_numpy(float)
    exog = panel[ARIMAX_COLS].to_numpy(float)
    fc, _ = arimax_forecast(precos, exog, exog[-1:], steps=1)
    return float(fc[-1])


def previsoes_por_modelo(horizon: int = 1) -> Dict[str, float]:
    """Reajusta cada modelo em toda a serie observada e preve h semanas a frente."""
    panel = load_panel(horizon)
    feat_cols = panel.attrs["feat_cols"]
    x_last, _, _ = latest_features(feat_cols)
    precos = _serie_precos()
    X = panel[feat_cols].to_numpy(float)
    y = panel["y"].to_numpy(float)

    out: Dict[str, float] = {}
    out["naive"] = float(naive_predict(precos, 1)[0])
    out["media_movel"] = float(ma_predict(precos, window=4, n=1)[0])
    try:
        fc, _ = arima_forecast(precos, steps=horizon)
        out["ARIMA"] = float(fc[-1])
    except Exception as exc:
        out["ARIMA"] = float("nan")
        print(f"  [previsao] ARIMA falhou: {exc}")
    try:
        out["ARIMAX"] = _prever_arimax(panel)
    except Exception as exc:
        out["ARIMAX"] = float("nan")
        print(f"  [previsao] ARIMAX falhou: {exc}")
    for nome, fit in (("LightGBM", fit_lgbm), ("XGBoost", fit_xgb)):
        try:
            modelo = fit(X, y)
            out[nome] = float(np.asarray(modelo.predict(x_last.reshape(1, -1))).ravel()[0])
        except Exception as exc:
            out[nome] = float("nan")
            print(f"  [previsao] {nome} falhou: {exc}")
    try:
        out["VS-ePL-KRLS"] = _prever_vsepl(panel, feat_cols, x_last)
    except Exception as exc:
        out["VS-ePL-KRLS"] = float("nan")
        print(f"  [previsao] VS-ePL-KRLS falhou: {exc}")
    return out


def ranking_h1() -> pd.DataFrame:
    tabela = pd.read_csv(RES / "semanal_benchmarks.csv")
    h1 = tabela[tabela.horizon == 1].sort_values("rmse").reset_index(drop=True)
    return h1


def escolher_vencedor(disponiveis: Dict[str, float]) -> tuple:
    """Menor RMSE em h=1 entre os modelos que sabem prever fora da amostra."""
    h1 = ranking_h1()
    descartados = []
    for _, row in h1.iterrows():
        nome = str(row["model"])
        if nome not in MODELOS_PRODUCAO:
            descartados.append(f"{nome} (sem previsao de producao)")
            continue
        if not np.isfinite(disponiveis.get(nome, np.nan)):
            descartados.append(f"{nome} (previsao nao finita)")
            continue
        return nome, float(row["rmse"]), descartados
    raise RuntimeError(f"Nenhum modelo elegivel no ranking h=1. Descartados: {descartados}")


def residuos_walkforward(modelo: str, horizon: int = 1) -> np.ndarray:
    path = RES / f"walkforward_preds_h{horizon}.csv"
    if not path.exists():
        return np.array([], dtype=float)
    wf = pd.read_csv(path)
    if modelo not in wf.columns:
        return np.array([], dtype=float)
    resid = wf["y"].astype(float) - wf[modelo].astype(float)
    resid = resid.to_numpy()
    return resid[np.isfinite(resid)]


def montar_previsao(horizon: int = 1) -> dict:
    """Payload completo da previsao publicada: vencedor, intervalo e alternativas."""
    panel = load_panel(horizon)
    feat_cols = panel.attrs["feat_cols"]
    _, ultima_data, ultimo_preco = latest_features(feat_cols)
    previsoes = previsoes_por_modelo(horizon)
    vencedor, rmse_wf, descartados = escolher_vencedor(previsoes)
    yhat = previsoes[vencedor]

    resid = residuos_walkforward(vencedor, horizon)[-JANELA_RESIDUOS:]
    if len(resid) >= 20:
        q10, q90 = (float(v) for v in np.quantile(resid, [0.10, 0.90]))
        p10, p90 = yhat + q10, yhat + q90
        probs = direction_probs(resid, yhat, ultimo_preco)
    else:
        p10 = p90 = None
        probs = {"p_alta": None, "p_estavel": None, "p_queda": None}

    semana_alvo = pd.Timestamp(ultima_data) + pd.Timedelta(weeks=horizon)
    ranking = ranking_h1()[["model", "rmse", "mae", "smape", "dir_acc"]]
    return {
        "modelo": vencedor,
        "criterio_selecao": "menor RMSE walk-forward em h=1",
        "rmse_walkforward": rmse_wf,
        "ultima_semana_observada": str(pd.Timestamp(ultima_data).date()),
        "preco_observado_ultima_semana": ultimo_preco,
        "semana_prevista": str(semana_alvo.date()),
        "horizonte": f"{horizon} semana" + ("s" if horizon > 1 else ""),
        "previsao_pontual": float(yhat),
        "p10": float(p10) if p10 is not None else None,
        "p90": float(p90) if p90 is not None else None,
        "probabilidades": probs,
        "n_residuos": int(len(resid)),
        "previsoes_por_modelo": {k: (float(v) if np.isfinite(v) else None) for k, v in previsoes.items()},
        "ranking_h1": ranking.to_dict(orient="records"),
        "modelos_descartados": descartados,
        "aviso": (
            "Previsao do preco medio nacional de REVENDA do Diesel B S-10 (ANP). "
            "Nao e preco de bomba de um posto especifico."
        ),
    }
