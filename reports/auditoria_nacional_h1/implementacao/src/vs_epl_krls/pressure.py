"""Pressao de repasse: quanto a refinaria esta atrasada em relacao a paridade.

O problema que isto ataca
-------------------------
O relatorio de paridade mediu o teto que sobra neste projeto: a variacao semanal
do preco de produtor correlaciona **+0,566** com a variacao seguinte da revenda,
o sinal mais forte ja encontrado aqui, mas a ANP publica o arquivo cerca de doze
dias depois do fim da semana de competencia.  Na defasagem que da para usar em
tempo real restam **+0,097**.  A conclusao registrada foi que o proximo ganho
material exigiria capturar os anuncios de reajuste da Petrobras no dia em que
saem.

Esses anuncios existem, sao publicos e trazem data e magnitude em R$/L — mas
saem como texto de assessoria de imprensa, sem serie baixavel.  Depender de
raspagem de portal para uma decisao de compra semanal e frágil.

A observacao que este modulo explora
------------------------------------
O anuncio nao e um evento aleatorio: ele e a *resposta* da refinaria a um desvio
acumulado em relacao a paridade de importacao.  E os dois lados desse desvio ja
estao no painel:

- o ultimo preco de produtor **conhecido**, defasado, mas conhecido;
- a paridade de importacao **de hoje**, diaria e em tempo real.

A distancia entre eles e observavel agora::

    pressao(T) = log(produtor(T - defasagem)) - log(paridade(T - 1))

Quando a paridade sobe e o produtor fica parado, essa distancia encolhe: a
refinaria esta vendendo barato em relacao ao custo de importar, e a probabilidade
de reajuste sobe.  Nao e o anuncio; e a condicao economica que o produz, e ela
nao precisa de raspagem nenhuma.

O nivel bruto nao serve como atributo — produtor e paridade tem bases de preco
diferentes, entao a diferenca tem um deslocamento arbitrario e lentamente
variavel.  O que entra e o desvio padronizado em janela expansiva, convertido
para R$/L pela multiplicacao pelo nivel de preco vigente, na mesma convencao dos
demais atributos do modelo de paridade.

Causalidade
-----------
Toda coluna devolvida aqui usa apenas informacao anterior a origem da previsao.
A padronizacao expansiva e deslocada em uma posicao para que uma observacao nunca
participe da propria normalizacao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PRESSURE_FEATURES", "build_pressure_features"]

#: Atributos de pressao, na mesma escala em R$/L dos atributos de paridade.
PRESSURE_FEATURES: tuple[str, ...] = ("press1", "dpress1")

#: Limiar pre-registrado do portao de decisao: pressao abaixo da propria media
#: historica expansiva.  Zero e escolha sem parametro ajustado — a refinaria
#: esta mais barata que o usual em relacao a paridade, e nada foi calibrado para
#: chegar a esse numero.
PRESSURE_GATE_Z = 0.0


def build_pressure_features(
    frame: pd.DataFrame,
    *,
    producer_lag: int = 3,
    warmup: int = 104,
) -> pd.DataFrame:
    """Colunas de pressao de repasse alinhadas por data.

    ``frame`` precisa das colunas produzidas por
    :func:`vs_epl_krls.causal_ingest.build_causal_panel`: ``date``, ``price``,
    ``parity`` e ``producer_price``.

    Devolve ``date``, ``press_z`` (o desvio padronizado puro, para uso como
    portao de decisao), ``press1`` (o mesmo desvio em R$/L, para uso como
    regressor) e ``dpress1`` (a variacao semanal).  Junte ao painel de paridade
    por ``date``.
    """

    required = {"date", "price", "parity", "producer_price"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pressure features require columns: {sorted(missing)}")
    if producer_lag < 1:
        raise ValueError("producer_lag must be at least one week")
    if warmup < 8:
        raise ValueError("warmup must cover at least eight weeks")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("price", "parity", "producer_price"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = (
        data.dropna(subset=["date", "price", "parity"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if (data["price"] <= 0).any() or (data["parity"] <= 0).any():
        raise ValueError("prices and parity must be strictly positive")

    producer = data["producer_price"].where(data["producer_price"] > 0)
    # O ultimo produtor publicado na origem, e a paridade da propria origem.
    known_producer = np.log(producer).shift(producer_lag)
    origin_parity = np.log(data["parity"]).shift(1)
    raw = known_producer - origin_parity

    # Padronizacao expansiva deslocada: a media e o desvio usados na semana T
    # foram calculados sem a semana T.
    mean = raw.expanding(min_periods=warmup).mean().shift(1)
    deviation = raw.expanding(min_periods=warmup).std().shift(1)
    standardized = (raw - mean) / deviation.replace(0.0, np.nan)

    previous_price = data["price"].shift(1)
    return pd.DataFrame(
        {
            "date": data["date"],
            "press_z": standardized,
            "press1": standardized * previous_price,
            "dpress1": standardized.diff() * previous_price,
        }
    )
