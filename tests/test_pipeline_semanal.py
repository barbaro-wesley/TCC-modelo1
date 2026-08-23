from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import forecast, weekly  # noqa: E402


def serie_semanal(n: int = 400, preco_final: float = 6.89) -> pd.DataFrame:
    # Ancorada na semana passada: assim a serie nunca cai no futuro, que e
    # justamente uma das reprovacoes de checar_sanidade.
    fim = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)
    datas = pd.date_range(end=fim, periods=n, freq="7D")
    precos = [3.5 + 0.01 * i for i in range(n - 1)] + [preco_final]
    return pd.DataFrame({"data": datas, "revenda": precos})


def estado_base(semanal: pd.DataFrame) -> dict:
    return {
        "n_linhas_semanal": len(semanal),
        "preco_ultima_semana": float(semanal["revenda"].iloc[-1]),
    }


def test_sanidade_aceita_serie_normal():
    semanal = serie_semanal()
    estado = estado_base(semanal)
    nova = pd.concat(
        [semanal, pd.DataFrame({"data": [semanal["data"].iloc[-1] + pd.Timedelta(weeks=1)], "revenda": [6.92]})],
        ignore_index=True,
    )
    assert weekly.checar_sanidade(nova, estado) == []


def test_sanidade_detecta_planilha_encolhida():
    semanal = serie_semanal()
    estado = estado_base(semanal)
    problemas = weekly.checar_sanidade(semanal.iloc[:-50], estado)
    assert any("encolheu" in p for p in problemas)


def test_sanidade_detecta_salto_de_preco():
    semanal = serie_semanal()
    estado = estado_base(semanal)
    quebrada = semanal.copy()
    quebrada.loc[quebrada.index[-1], "revenda"] = 68.9  # virgula deslocada na origem
    problemas = weekly.checar_sanidade(quebrada, estado)
    assert any("salto de preco" in p for p in problemas)


def test_sanidade_detecta_data_no_futuro():
    semanal = serie_semanal()
    estado = estado_base(semanal)
    futura = semanal.copy()
    futura.loc[futura.index[-1], "data"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=60)
    problemas = weekly.checar_sanidade(futura, estado)
    assert any("futuro" in p for p in problemas)


def test_vencedor_ignora_modelo_sem_previsao_de_producao(monkeypatch):
    ranking = pd.DataFrame(
        {
            "model": ["LSTM", "VS-ePL-KRLS", "ARIMA"],
            "rmse": [0.01, 0.02, 0.07],
        }
    )
    monkeypatch.setattr(forecast, "ranking_h1", lambda: ranking)
    previsoes = {"ARIMA": 6.88, "VS-ePL-KRLS": float("nan")}
    nome, rmse, descartados = forecast.escolher_vencedor(previsoes)
    assert nome == "ARIMA"
    assert rmse == 0.07
    assert len(descartados) == 2  # LSTM (sem producao) e VS-ePL-KRLS (nao finita)


def test_historico_preenche_realizado_da_semana_passada(monkeypatch):
    tmp = tempfile.TemporaryDirectory()
    monkeypatch.setattr(weekly, "HISTORICO", Path(tmp.name) / "historico.json")
    semanal = pd.DataFrame(
        {
            "data": pd.to_datetime(["2026-08-09", "2026-08-16"]),
            "revenda": [6.91, 6.89],
        }
    )
    antiga = {
        "modelo": "ARIMA",
        "criterio_selecao": "menor RMSE walk-forward em h=1",
        "ultima_semana_observada": "2026-08-09",
        "preco_observado_ultima_semana": 6.91,
        "semana_prevista": "2026-08-16",
        "previsao_pontual": 6.90,
        "p10": 6.85,
        "p90": 6.95,
    }
    weekly.atualizar_historico(antiga, semanal)

    nova = dict(antiga, ultima_semana_observada="2026-08-16", semana_prevista="2026-08-23",
                preco_observado_ultima_semana=6.89, previsao_pontual=6.88)
    registros = weekly.atualizar_historico(nova, semanal)

    por_semana = {r["semana_prevista"]: r for r in registros}
    fechada = por_semana["2026-08-16"]
    assert fechada["preco_realizado"] == 6.89
    assert abs(fechada["erro"] - (6.89 - 6.90)) < 1e-9
    # a semana ainda nao publicada fica sem realizado
    assert por_semana["2026-08-23"]["preco_realizado"] is None
    tmp.cleanup()
