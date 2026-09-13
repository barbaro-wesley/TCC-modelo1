"""Modelo estadual do Diesel B S10, com foco operacional no Rio Grande do Sul.

Por que estadual
---------------
O produto previa o preco medio **nacional** de revenda da ANP.  Nenhum comprador
paga esse preco: ele e uma media de 3.173 postos espalhados por 27 unidades da
federacao, com carga tributaria, frete e estrutura de distribuicao diferentes em
cada uma.  A distancia entre a serie modelada e a serie que o cliente enfrenta
era a maior fragilidade comercial do projeto, e ela nao se resolve com modelo
melhor: resolve-se com o dado certo.

A ANP publica a mesma pesquisa semanal desagregada por estado.  Para o Rio Grande
do Sul sao 702 semanas ininterruptas desde 2012-12-30, com mediana de 262 postos
pesquisados por semana.

Decomposicao, e por que ela e melhor que modelar o estado direto
----------------------------------------------------------------
Medido sobre as 702 semanas comuns:

- a variacao semanal do RS correlaciona **0,939** com a nacional, ou seja,
  **88% da variancia do estado e movimento do pais**;
- o desvio da variacao semanal e 0,0769 no RS contra 0,0735 no Brasil — o estado
  e *mais* ruidoso, o que faz sentido com 262 postos contra 3.173;
- o desvio da variacao do **spread** RS menos Brasil e apenas 0,0264.

Modelar o RS diretamente joga fora o sinal nacional, que e melhor medido, e
paga o ruido estadual inteiro.  A decomposicao ``rs = brasil + spread`` mantem a
maquinaria nacional onde ela e forte e isola um residuo pequeno e bem comportado::

    rs(T) = brasil(T) + spread(T)

O spread reverte a media com meia-vida de cerca de **20 semanas**
(autocorrelacao semanal 0,961), o que o torna previsivel de um jeito que o preco
em nivel nao e.

Ancora causal regional
----------------------
O arquivo de precos de produtor da ANP e publicado por regiao.  Para o RS a
regiao certa e **Sul**, servida pela REFAP em Canoas, e ela e mensuravelmente
melhor que a mediana nacional usada ate aqui: a variacao do produtor Sul em
defasagem de uma semana correlaciona **+0,4265** com a variacao seguinte da
revenda gaucha, contra +0,3952 da mediana entre regioes.

O spread de produtor (Sul menos nacional) explica o **nivel** do spread de
revenda — correlacao estavel de +0,25 em varias defasagens — mas quase nada da
variacao semanal dele (+0,06).  Ele entra, portanto, como ancora do alvo de
reversao, nao como preditor de curto prazo.  A distincao importa: usar um sinal
de nivel como se fosse de variacao e o tipo de erro que produz backtest bonito e
producao ruim.

Causalidade
-----------
Tudo aqui e estimado em janela expansiva com informacao anterior a origem.  O
preco de produtor entra apenas com a defasagem de publicacao da ANP.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import io
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .causal_ingest import (
    ANP_PRODUCER_URL,
    PRODUCER_PRODUCT,
    PRODUCER_PUBLICATION_LAG_WEEKS,
    SourceRecord,
    _get,
    _record,
    _REGIONS,
)

__all__ = [
    "ANP_STATES_URL",
    "UF_REGION",
    "PooledReversion",
    "SpreadConfig",
    "SpreadForecast",
    "SpreadForecaster",
    "build_regional_panel",
    "extract_state_series",
    "fetch_regional_producer",
    "fetch_state_weekly",
    "fetch_states_workbook",
    "normalize_label",
    "parse_states_workbook",
    "pool_reversion",
]

ANP_STATES_URL = (
    "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/"
    "precos-revenda-e-de-distribuicao-combustiveis/shlp/semanal/"
    "semanal-estados-desde-2013.xlsx"
)

#: Cabecalho real da planilha estadual da ANP.  O arquivo traz dezessete linhas
#: de nota antes da tabela; o numero e verificado na leitura.
_STATES_HEADER_ROW = 17

#: Regiao da ANP a que cada unidade da federacao pertence.  E o que define qual
#: preco de produtor e o insumo causal correto para aquele estado.
UF_REGION: dict[str, str] = {
    "AC": "norte", "AP": "norte", "AM": "norte", "PA": "norte",
    "RO": "norte", "RR": "norte", "TO": "norte",
    "AL": "nordeste", "BA": "nordeste", "CE": "nordeste", "MA": "nordeste",
    "PB": "nordeste", "PE": "nordeste", "PI": "nordeste", "RN": "nordeste",
    "SE": "nordeste",
    "DF": "centro_oeste", "GO": "centro_oeste", "MT": "centro_oeste",
    "MS": "centro_oeste",
    "ES": "sudeste", "MG": "sudeste", "RJ": "sudeste", "SP": "sudeste",
    "PR": "sul", "RS": "sul", "SC": "sul",
}

_UF_NAME: dict[str, str] = {
    "AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAPA", "AM": "AMAZONAS",
    "BA": "BAHIA", "CE": "CEARA", "DF": "DISTRITO FEDERAL",
    "ES": "ESPIRITO SANTO", "GO": "GOIAS", "MA": "MARANHAO",
    "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS",
    "PA": "PARA", "PB": "PARAIBA", "PR": "PARANA", "PE": "PERNAMBUCO",
    "PI": "PIAUI", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE",
    "RS": "RIO GRANDE DO SUL", "RO": "RONDONIA", "RR": "RORAIMA",
    "SC": "SANTA CATARINA", "SP": "SAO PAULO", "SE": "SERGIPE",
    "TO": "TOCANTINS",
}


def normalize_label(value: object) -> str:
    """Maiusculas sem acento, para casar rotulos da planilha da ANP."""

    text = str(value).strip().upper()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).replace("Ç", "C")


def fetch_states_workbook(cache: Path | None = None) -> bytes:
    """Baixa a planilha estadual uma vez.

    Sao 12,5 MB que contem as 27 unidades da federacao.  Baixar por estado seria
    pagar isso 27 vezes para ler a mesma tabela.
    """

    payload = _get(ANP_STATES_URL)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    return payload


def parse_states_workbook(payload: bytes) -> pd.DataFrame:
    """Le a planilha estadual uma vez e valida o formato."""

    raw = pd.read_excel(io.BytesIO(payload), sheet_name=0, header=_STATES_HEADER_ROW)
    raw.columns = [normalize_label(column) for column in raw.columns]
    required = {"DATA INICIAL", "ESTADO", "PRODUTO", "UNIDADE DE MEDIDA", "PRECO MEDIO REVENDA"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"planilha estadual da ANP mudou de formato; faltam {sorted(missing)}")
    return raw


def extract_state_series(raw: pd.DataFrame, uf: str) -> pd.DataFrame:
    """Extrai a serie de Diesel S10 de uma unidade da federacao ja lida.

    Devolve ``date`` (domingo que inicia a semana pesquisada), ``price`` em R$/L
    e ``stations``, o numero de postos que sustentam aquela media — informacao
    de qualidade que o agregado nacional escondia.
    """

    code = str(uf).strip().upper()
    if code not in _UF_NAME:
        raise ValueError(f"unidade da federacao desconhecida: {uf!r}")

    state = raw[raw["ESTADO"].map(normalize_label) == _UF_NAME[code]].copy()
    product = state["PRODUTO"].map(normalize_label)
    state = state[product.str.contains("DIESEL") & product.str.contains("S10|S-10", regex=True)]
    if state.empty:
        raise ValueError(f"nenhuma linha de Diesel S10 para {code} na planilha estadual")
    units = {normalize_label(u) for u in state["UNIDADE DE MEDIDA"].dropna()}
    if units != {"R$/L"}:
        raise ValueError(f"unidade inesperada na serie estadual: {sorted(units)}")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(state["DATA INICIAL"], errors="coerce"),
            "price": pd.to_numeric(state["PRECO MEDIO REVENDA"], errors="coerce"),
            "stations": pd.to_numeric(
                state.get("NUMERO DE POSTOS PESQUISADOS"), errors="coerce"
            ),
        }
    )
    frame = (
        frame.dropna(subset=["date", "price"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if (frame["price"] <= 0).any():
        raise ValueError("precos estaduais precisam ser estritamente positivos")
    return frame


def fetch_state_weekly(
    uf: str, *, cache: Path | None = None
) -> tuple[pd.DataFrame, SourceRecord]:
    """Baixa e extrai a serie de um unico estado, com proveniencia."""

    # Valida antes de baixar: nao faz sentido puxar 12,5 MB para descobrir que a
    # sigla nao existe.
    if str(uf).strip().upper() not in _UF_NAME:
        raise ValueError(f"unidade da federacao desconhecida: {uf!r}")
    payload = fetch_states_workbook(cache)
    frame = extract_state_series(parse_states_workbook(payload), uf)
    return frame, _record(
        f"anp_estado_{str(uf).strip().lower()}",
        ANP_STATES_URL,
        payload,
        frame,
        frame["date"],
    )


def fetch_regional_producer(
    region: str, *, cache: Path | None = None
) -> tuple[pd.DataFrame, SourceRecord]:
    """Preco de produtor da regiao pedida e o agregado nacional, lado a lado.

    O agregado nacional continua sendo a **mediana entre regioes**: a coluna
    ``Brasil`` do arquivo oficial e defeituosa em varias semanas, marcando valores
    que excedem todas as regioes menos uma.
    """

    key = str(region).strip().lower()
    if key not in _REGIONS:
        raise ValueError(f"regiao desconhecida: {region!r}; use uma de {_REGIONS}")

    payload = _get(ANP_PRODUCER_URL)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    raw = pd.read_excel(io.BytesIO(payload), header=None, skiprows=9)
    raw.columns = ["produto", "ini", "fim", *_REGIONS, "brasil", "extra"][: raw.shape[1]]
    subset = raw[raw["produto"] == PRODUCER_PRODUCT].copy()
    if subset.empty:
        raise ValueError(f"produto ausente no arquivo de produtores: {PRODUCER_PRODUCT}")
    subset["date"] = pd.to_datetime(subset["ini"], errors="coerce")
    for column in _REGIONS:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset["producer_national"] = subset[list(_REGIONS)].median(axis=1)
    subset = subset.dropna(subset=["date", key, "producer_national"])
    if (subset[key] <= 0).any():
        raise ValueError("precos de produtor precisam ser estritamente positivos")

    frame = (
        subset[["date", key, "producer_national"]]
        .rename(columns={key: "producer_region"})
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    return frame, _record(
        f"anp_producer_{key}", ANP_PRODUCER_URL, payload, frame, frame["date"]
    )


def build_regional_panel(
    state: pd.DataFrame,
    national: pd.DataFrame,
    producer: pd.DataFrame,
    *,
    producer_lag: int = PRODUCER_PUBLICATION_LAG_WEEKS,
    warmup: int = 104,
) -> pd.DataFrame:
    """Painel do estado alinhado ao nacional, com o spread e suas âncoras.

    ``state`` precisa de ``date`` e ``price``; ``national`` de ``date`` e
    ``price``; ``producer`` de ``date``, ``producer_region`` e
    ``producer_national``.  A semana de competencia do arquivo de produtor comeca
    um dia depois da semana de revenda, entao o alinhamento e por proximidade com
    tolerancia de tres dias, nunca por posicao.
    """

    for name, frame, columns in (
        ("state", state, {"date", "price"}),
        ("national", national, {"date", "price"}),
        ("producer", producer, {"date", "producer_region", "producer_national"}),
    ):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} precisa das colunas {sorted(missing)}")
    if producer_lag < 1:
        raise ValueError("producer_lag precisa ser ao menos uma semana")

    merged = (
        state[["date", "price"] + (["stations"] if "stations" in state else [])]
        .rename(columns={"price": "state_price"})
        .merge(
            national[["date", "price"]].rename(columns={"price": "national_price"}),
            on="date",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    if merged.empty:
        raise ValueError("estado e nacional nao tem semanas em comum")

    merged = pd.merge_asof(
        merged,
        producer.sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3),
    )

    panel = pd.DataFrame({"date": merged["date"]})
    if "stations" in merged:
        panel["stations"] = merged["stations"]
    panel["price"] = merged["state_price"]
    panel["national_price"] = merged["national_price"]
    panel["origin_price"] = merged["state_price"].shift(1)
    panel["spread"] = merged["state_price"] - merged["national_price"]
    panel["y"] = panel["spread"].diff()
    panel["spread_lag1"] = panel["spread"].shift(1)
    panel["dspread1"] = panel["y"].shift(1)

    # Ancora de nivel: o spread de produtor da regiao contra o pais, conhecido
    # apenas com a defasagem de publicacao da ANP.
    producer_spread = np.log(merged["producer_region"]) - np.log(merged["producer_national"])
    panel["producer_spread"] = producer_spread.shift(producer_lag)
    mean = panel["producer_spread"].expanding(min_periods=warmup).mean().shift(1)
    deviation = panel["producer_spread"].expanding(min_periods=warmup).std().shift(1)
    panel["producer_spread_z"] = (panel["producer_spread"] - mean) / deviation.replace(0.0, np.nan)
    return panel


@dataclass(frozen=True)
class PooledReversion:
    """Reversao do conjunto dos estados e o encolhimento de cada um em direcao a ela."""

    pooled_kappa: float
    between_variance: float
    heterogeneity_q: float
    n_states: int
    #: Peso do proprio estado na mistura.  Perto de 1: o estado sabe o suficiente
    #: sobre si mesmo.  Perto de 0: adota a reversao do conjunto.
    weights: dict[str, float]
    shrunk_kappa: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def pool_reversion(
    estimates: Mapping[str, tuple[float, float]],
    *,
    min_states: int = 3,
) -> PooledReversion:
    """Encolhe a reversao de cada estado na direcao do conjunto.

    Por que isto existe
    -------------------
    Sao 27 series com o mesmo mecanismo economico e tamanhos de amostra muito
    diferentes: o Rio Grande do Sul tem mediana de 262 postos pesquisados por
    semana, e estados pequenos tem uma fracao disso.  A reversao estimada num
    estado pequeno e, em boa parte, ruido — mas jogar fora a estimativa local e
    tao errado quanto confiar nela inteira.

    O estimador e o de efeitos aleatorios de DerSimonian e Laird (1986).  Ele
    separa a variancia *entre* estados, ``tau^2``, da incerteza *dentro* de cada
    estimativa, ``se^2``, e devolve o peso classico de Bayes empirico::

        peso(uf) = tau^2 / (tau^2 + se(uf)^2)

    A leitura e direta: quando os estados de fato diferem entre si, ``tau^2`` e
    grande e cada um fica com o proprio numero; quando a diferenca observada
    cabe dentro do erro de estimativa, ``tau^2`` colapsa e todos convergem para o
    valor comum.  Nenhum limiar arbitrario decide isso — os dados decidem.

    ``estimates`` mapeia a sigla do estado para ``(kappa, erro_padrao)``.
    Estimativas sem erro-padrao finito e positivo sao descartadas: sem elas nao
    ha como pesar nada.
    """

    usable = {
        str(uf).strip().upper(): (float(kappa), float(error))
        for uf, (kappa, error) in estimates.items()
        if np.isfinite(kappa) and np.isfinite(error) and error > 0
    }
    if len(usable) < min_states:
        # Sem estados suficientes nao ha conjunto de onde tomar emprestado; cada
        # um fica com o proprio numero, declaradamente.
        return PooledReversion(
            pooled_kappa=float("nan"),
            between_variance=float("nan"),
            heterogeneity_q=float("nan"),
            n_states=len(usable),
            weights={uf: 1.0 for uf in usable},
            shrunk_kappa={uf: value for uf, (value, _) in usable.items()},
        )

    codes = list(usable)
    kappa = np.array([usable[uf][0] for uf in codes], dtype=float)
    variance = np.array([usable[uf][1] ** 2 for uf in codes], dtype=float)

    fixed_weights = 1.0 / variance
    fixed_mean = float(np.sum(fixed_weights * kappa) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (kappa - fixed_mean) ** 2))
    s1 = float(np.sum(fixed_weights))
    s2 = float(np.sum(fixed_weights**2))
    denominator = s1 - s2 / s1
    between = max(0.0, (q - (len(codes) - 1)) / denominator) if denominator > 0 else 0.0

    random_weights = 1.0 / (variance + between)
    pooled = float(np.sum(random_weights * kappa) / np.sum(random_weights))
    weights = {
        uf: (between / (between + variance[index]) if between + variance[index] > 0 else 0.0)
        for index, uf in enumerate(codes)
    }
    return PooledReversion(
        pooled_kappa=pooled,
        between_variance=between,
        heterogeneity_q=q,
        n_states=len(codes),
        weights=weights,
        shrunk_kappa={
            uf: weights[uf] * usable[uf][0] + (1.0 - weights[uf]) * pooled for uf in codes
        },
    )


@dataclass(frozen=True)
class SpreadConfig:
    """Parametros do modelo de spread, todos declarados e nenhum ajustado."""

    #: Minimo de semanas antes de estimar; abaixo disso o modelo carrega o spread.
    min_train: int = 104
    #: Peso maximo da ancora de produtor sobre o alvo de reversao.  Limitado
    #: porque a ancora explica nivel, nao variacao: deixa-la livre convidaria o
    #: ajuste a persegui-la em alta frequencia.
    anchor_weight_limit: float = 1.0
    #: Reversao maxima por semana.  Meia-vida medida e de ~20 semanas; um phi
    #: implausivelmente baixo indicaria ajuste dominado por poucos pontos.
    max_reversion: float = 0.5
    #: Limites do peso por numero de postos, em multiplos da mediana do estado.
    #: Sem eles uma semana de amostra minima somem do ajuste e uma de amostra
    #: excepcional passa a mandar sozinha.
    min_station_weight: float = 0.25
    max_station_weight: float = 4.0
    #: Limite de variacao prevista, em desvios do proprio spread.  Ele **limita**
    #: a magnitude; nao cancela a direcao.  Zerar a reversao ao ultrapassar o
    #: limite silenciaria o modelo exatamente no extremo, que e onde o sinal de
    #: reversao e mais forte — o oposto do comportamento desejado.
    change_limit_sigma: float = 4.0


@dataclass(frozen=True)
class SpreadForecast:
    """Previsao do spread e do preco estadual dela derivado."""

    target_date: pd.Timestamp
    origin_spread: float
    spread_point: float
    spread_change: float
    national_point: float
    state_point: float
    reversion: float
    anchor_target: float
    fallback_used: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_date"] = str(pd.Timestamp(self.target_date).date())
        return payload


@dataclass
class SpreadForecaster:
    """Correcao de erro sobre o spread estado menos nacional.

    Especificacao, com o spread ``s`` e a ancora padronizada ``a`` do produtor::

        s(T) - s(T-1) = -kappa * [ s(T-1) - (mu + lambda * a(T-1)) ]

    Em portugues: o spread anda de volta na direcao de um alvo, e o alvo nao e
    uma constante — ele desliza com a posicao relativa do preco de produtor da
    regiao.  Quando a ancora nao esta disponivel, ``lambda`` some e o alvo vira a
    media expansiva, que e o caso base honesto.

    A estimacao e por minimos quadrados em janela expansiva.  Nao ha Huber aqui
    de proposito: o spread nao tem a cauda do preco em nivel, e o proprio
    experimento mediu desvio de 0,0264 na variacao semanal contra 0,0769 do
    preco.  Robustez que nao e necessaria custa eficiencia.
    """

    config: SpreadConfig = field(default_factory=SpreadConfig)
    use_anchor: bool = True
    #: Reversao estimada no conjunto dos estados.  Quando informada, a reversao
    #: local e encolhida na direcao dela por ``pooling_weight``.
    pooled_kappa: float | None = None
    #: Peso do proprio estado na mistura, entre zero e um.  Um estado com muitos
    #: postos pesquisados fica perto de 1 e quase nao toma emprestado; um estado
    #: pequeno fica perto de 0 e adota a reversao do conjunto.
    pooling_weight: float = 1.0
    #: Pondera cada semana pelo numero de postos que a ANP pesquisou nela.
    #:
    #: A media semanal de um estado e uma media amostral, e a ANP publica o
    #: tamanho da amostra.  Medido nos dez estados do experimento: as semanas do
    #: quartil inferior de postos tem cerca de **1,9x** a volatilidade do spread
    #: das demais.  Tratar uma semana de 8 postos como uma de 200 e descartar
    #: informacao que o proprio arquivo entrega.
    weight_by_stations: bool = False
    kappa_: float = 0.0
    kappa_local_: float = 0.0
    kappa_se_: float = float("nan")
    mu_: float = 0.0
    lambda_: float = 0.0
    sigma_: float = 0.0
    n_train_: int = 0

    def fit(self, panel: pd.DataFrame, *, end: int | None = None) -> "SpreadForecaster":
        end = len(panel) if end is None else int(end)
        if end <= 0 or end > len(panel):
            raise ValueError("end esta fora do painel")
        window = panel.iloc[:end]
        columns = ["y", "spread_lag1"] + (["producer_spread_z"] if self.use_anchor else [])
        use_stations = self.weight_by_stations and "stations" in window.columns
        if use_stations:
            columns = columns + ["stations"]
        usable = window[columns].dropna()
        if len(usable) < self.config.min_train:
            raise ValueError(
                f"historico insuficiente para ajustar o spread: {len(usable)} < {self.config.min_train}"
            )
        target = usable["y"].to_numpy(float)
        lagged = usable["spread_lag1"].to_numpy(float)
        design = [np.ones(len(usable)), lagged]
        if self.use_anchor:
            design.append(usable["producer_spread_z"].to_numpy(float))
        matrix = np.column_stack(design)

        if use_stations:
            # Minimos quadrados ponderados: a variancia de uma media amostral cai
            # com o tamanho da amostra, entao o peso proporcional ao numero de
            # postos e o peso estatisticamente correto.  Os limites impedem que
            # uma semana isolada domine ou desapareca.
            stations = usable["stations"].to_numpy(float)
            weights = stations / max(float(np.median(stations)), 1.0)
            weights = np.clip(weights, self.config.min_station_weight, self.config.max_station_weight)
            root = np.sqrt(weights)[:, None]
            beta, *_ = np.linalg.lstsq(matrix * root, target * root.ravel(), rcond=None)
        else:
            beta, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        if not np.all(np.isfinite(beta)):
            raise ValueError("a regressao do spread nao convergiu")

        # Erro-padrao do coeficiente de reversao.  E ele que diz o quanto este
        # estado sabe sobre a propria reversao, e portanto o quanto ele deveria
        # tomar emprestado do conjunto.
        residual_for_se = target - matrix @ beta
        dof = max(len(usable) - matrix.shape[1], 1)
        variance = float(residual_for_se @ residual_for_se) / dof
        try:
            covariance = np.linalg.inv(matrix.T @ matrix) * variance
            kappa_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
        except np.linalg.LinAlgError:  # pragma: no cover - matriz singular
            kappa_se = float("nan")
        self.kappa_se_ = kappa_se

        kappa = float(-beta[1])
        kappa = float(np.clip(kappa, 0.0, self.config.max_reversion))
        self.kappa_local_ = kappa
        if self.pooled_kappa is not None:
            weight = float(np.clip(self.pooling_weight, 0.0, 1.0))
            kappa = float(
                np.clip(
                    weight * kappa + (1.0 - weight) * float(self.pooled_kappa),
                    0.0,
                    self.config.max_reversion,
                )
            )
        if kappa <= 0.0:
            # Sem reversao estimavel, o melhor palpite honesto e carregar o
            # spread atual: nao inventar retorno a media que os dados nao mostram.
            self.kappa_, self.mu_, self.lambda_ = 0.0, float(np.mean(lagged)), 0.0
        else:
            self.kappa_ = kappa
            self.mu_ = float(beta[0] / kappa)
            if self.use_anchor:
                weight = float(beta[2] / kappa)
                limit = self.config.anchor_weight_limit
                self.lambda_ = float(np.clip(weight, -limit, limit))
            else:
                self.lambda_ = 0.0
        residual = target - matrix @ beta
        self.sigma_ = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
        self.n_train_ = int(len(usable))
        return self

    def _target(self, anchor: float) -> float:
        if self.use_anchor and np.isfinite(anchor):
            return self.mu_ + self.lambda_ * anchor
        return self.mu_

    def forecast_row(
        self, row: pd.Series, *, national_point: float
    ) -> SpreadForecast:
        """Previsao para uma linha do painel, com fallback para carregar o spread."""

        if not np.isfinite(national_point) or national_point <= 0:
            raise ValueError("national_point precisa ser finito e positivo")
        origin = float(row.get("spread_lag1", np.nan))
        fallback, reason = False, ""
        if not np.isfinite(origin):
            origin = 0.0
            fallback, reason = True, "spread_indisponivel"
        anchor = float(row.get("producer_spread_z", np.nan))
        target = self._target(anchor)
        change = self.kappa_ * (target - origin)
        limit = self.config.change_limit_sigma * (self.sigma_ or np.inf)
        if np.isfinite(limit) and abs(change) > limit:
            change = float(np.sign(change) * limit)
            fallback = True
            reason = reason or "variacao_limitada"
        point = origin + change
        return SpreadForecast(
            target_date=row.get("date", pd.NaT),
            origin_spread=origin,
            spread_point=float(point),
            spread_change=float(change),
            national_point=float(national_point),
            state_point=float(national_point + point),
            reversion=float(self.kappa_),
            anchor_target=float(target),
            fallback_used=fallback,
            reason=reason,
        )

    def walk_forward(
        self,
        panel: pd.DataFrame,
        national_predictions: NDArray[np.float64],
        start: int,
        end: int,
        *,
        refit_every: int = 1,
    ) -> pd.DataFrame:
        """Reajusta em janela expansiva e preve ``[start, end)`` sem olhar o futuro."""

        if not 0 < start < end <= len(panel):
            raise ValueError("janela de walk-forward invalida")
        if len(national_predictions) != end - start:
            raise ValueError("as previsoes nacionais nao cobrem a janela pedida")
        # Propaga o pooling: sem isto o modelo interno do walk-forward recomeca
        # sem o conjunto e a avaliacao mede sempre o ajuste local.
        model = SpreadForecaster(
            config=self.config,
            use_anchor=self.use_anchor,
            pooled_kappa=self.pooled_kappa,
            pooling_weight=self.pooling_weight,
            weight_by_stations=self.weight_by_stations,
        )
        records: list[dict[str, object]] = []
        last_fit = -(10**9)
        fitted = False
        for offset, index in enumerate(range(start, end)):
            if index - last_fit >= refit_every:
                try:
                    model.fit(panel, end=index)
                    last_fit, fitted = index, True
                except ValueError:
                    pass
            row = panel.iloc[index]
            national_point = float(national_predictions[offset])
            if not fitted or not np.isfinite(national_point):
                records.append(
                    {
                        "date": row["date"],
                        "actual": float(row["price"]),
                        "prediction": np.nan,
                        "persistence": float(row.get("origin_price", np.nan)),
                        "spread_point": np.nan,
                        "fallback": True,
                    }
                )
                continue
            forecast = model.forecast_row(row, national_point=national_point)
            records.append(
                {
                    "date": row["date"],
                    "actual": float(row["price"]),
                    "prediction": forecast.state_point,
                    "persistence": float(row.get("origin_price", np.nan)),
                    "spread_point": forecast.spread_point,
                    "fallback": forecast.fallback_used,
                }
            )
        return pd.DataFrame.from_records(records)

    def summary(self) -> dict[str, object]:
        if not self.n_train_:
            return {"fitted": False}
        half_life = float(np.log(2) / -np.log(1 - self.kappa_)) if 0 < self.kappa_ < 1 else None
        return {
            "fitted": True,
            "n_train": self.n_train_,
            "reversion_kappa": round(self.kappa_, 6),
            "reversion_kappa_local": round(self.kappa_local_, 6),
            "reversion_kappa_se": (
                None if not np.isfinite(self.kappa_se_) else round(self.kappa_se_, 6)
            ),
            "pooled_kappa": (
                None if self.pooled_kappa is None else round(float(self.pooled_kappa), 6)
            ),
            "pooling_weight": round(float(self.pooling_weight), 6),
            "half_life_weeks": None if half_life is None else round(half_life, 2),
            "target_mean": round(self.mu_, 6),
            "anchor_weight": round(self.lambda_, 6),
            "residual_sigma": round(self.sigma_, 6),
            "uses_producer_anchor": self.use_anchor,
            "weighted_by_stations": self.weight_by_stations,
        }
