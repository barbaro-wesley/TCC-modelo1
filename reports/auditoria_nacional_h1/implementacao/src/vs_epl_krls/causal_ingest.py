"""Ingestao automatizada das fontes causais do Diesel B S10, com proveniencia.

Fontes e por que cada uma esta aqui
-----------------------------------
``anp_producer``
    Precos medios ponderados semanais praticados por produtores e importadores
    (ANP).  E o preco de refinaria do Diesel S-10, o driver direto do preco de
    bomba.  **Restricao critica:** o proprio arquivo declara que a atualizacao
    ocorre cerca de doze dias apos o encerramento da semana de competencia.  Com
    a previsao feita quando a revenda da semana anterior e publicada, isso torna
    o dado utilizavel apenas com tres semanas de defasagem — e nessa defasagem o
    sinal ja perdeu quase toda a forca (correlacao com a variacao da revenda cai
    de +0,57 em lag 1 para +0,10 em lag 3).  Ele entra no painel documentado,
    mas nao carrega o modelo.

``ulsd``
    Futuro de ULSD (NY Harbor, simbolo ``HO=F``), em USD/galao.  E o benchmark
    *de diesel*, muito mais proximo do produto que o Brent, que e petroleo cru.
    Substitui a fonte anterior do repositorio, que retornava uma pagina de
    verificacao de robo e produzia uma coluna 100% nula em silencio.

``brent`` e ``usdbrl``
    Series diarias do IPEA, ja usadas pelo repositorio.  Combinadas com o ULSD
    formam a paridade de importacao em R$/L.

Alinhamento temporal
--------------------
O indice semanal da ANP e datado pelo domingo que *inicia* a semana pesquisada.
Toda serie diaria e projetada nesse indice por ``ffill``, de modo que o valor na
linha ``T`` e o ultimo fechamento anterior ao inicio da semana ``T`` — estritamente
no passado de toda a janela de medicao do preco de revenda.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests  # type: ignore[import-untyped]

__all__ = [
    "SourceRecord",
    "PRODUCER_PUBLICATION_LAG_WEEKS",
    "fetch_ulsd",
    "fetch_producer_prices",
    "fetch_ipea_series",
    "build_causal_panel",
]

_TIMEOUT = 180
_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
_GALLON_LITERS = 3.785411784

#: A ANP publica o arquivo de produtores ~12 dias apos o fim da semana de
#: competencia.  Com a previsao emitida logo apos a revenda da semana anterior
#: sair, apenas defasagens de tres semanas ou mais estao efetivamente
#: disponiveis.  Este numero e uma restricao operacional, nao um ajuste.
PRODUCER_PUBLICATION_LAG_WEEKS = 3

ANP_PRODUCER_URL = (
    "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/precos/"
    "ppidp/precos-medios-ponderados-semanais-2013.xls"
)
YAHOO_ULSD_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/HO=F"
    "?period1=1293840000&period2=9999999999&interval=1d"
)
IPEA_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{code}')"
PRODUCER_PRODUCT = "Óleo Diesel S-10 (R$/litro)"
_REGIONS = ("norte", "nordeste", "centro_oeste", "sul", "sudeste")


@dataclass(frozen=True)
class SourceRecord:
    """Proveniencia de uma fonte baixada."""

    name: str
    url: str
    sha256: str
    n_bytes: int
    fetched_at: str
    rows: int
    coverage_start: str
    coverage_end: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get(url: str) -> bytes:
    response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.content


def _record(name: str, url: str, payload: bytes, frame: pd.DataFrame | pd.Series,
            dates: pd.Series) -> SourceRecord:
    return SourceRecord(
        name=name,
        url=url,
        sha256=hashlib.sha256(payload).hexdigest(),
        n_bytes=len(payload),
        fetched_at=_now(),
        rows=int(len(frame)),
        coverage_start=str(pd.Timestamp(dates.min()).date()),
        coverage_end=str(pd.Timestamp(dates.max()).date()),
    )


def fetch_ulsd(cache: Path | None = None) -> tuple[pd.Series, SourceRecord]:
    """Fechamentos diarios do futuro de ULSD em USD/galao."""

    payload = _get(YAHOO_ULSD_URL)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    document = json.loads(payload)
    result = document["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s").normalize(),
            "ulsd": pd.to_numeric(closes, errors="coerce"),
        }
    ).dropna()
    frame = frame.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if frame.empty or (frame["ulsd"] <= 0).any():
        raise ValueError("ULSD series came back empty or non-positive")
    series = frame.set_index("date")["ulsd"]
    return series, _record("ulsd", YAHOO_ULSD_URL, payload, frame, frame["date"])


def fetch_producer_prices(cache: Path | None = None) -> tuple[pd.DataFrame, SourceRecord]:
    """Preco semanal do Diesel S-10 no produtor, agregado de forma robusta.

    A coluna ``Brasil`` do arquivo apresenta valores inconsistentes em 2026 —
    em varias semanas ela excede todas as regioes menos uma, o que e impossivel
    para uma media ponderada, e o proprio arquivo declara que os dados sao
    preliminares.  Usamos a **mediana entre regioes**, que se mostrou o agregado
    mais estavel (desvio das variacoes semanais 0,114 contra 0,140 da coluna
    Brasil) e imune a esses episodios.
    """

    payload = _get(ANP_PRODUCER_URL)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(payload)
    raw = pd.read_excel(io.BytesIO(payload), header=None, skiprows=9)
    raw.columns = ["produto", "ini", "fim", *_REGIONS, "brasil", "extra"][: raw.shape[1]]
    subset = raw[raw["produto"] == PRODUCER_PRODUCT].copy()
    if subset.empty:
        raise ValueError(f"product not found in the ANP producer file: {PRODUCER_PRODUCT}")
    subset["ini"] = pd.to_datetime(subset["ini"], errors="coerce")
    for column in (*_REGIONS, "brasil"):
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna(subset=["ini"]).sort_values("ini").reset_index(drop=True)
    subset["producer_price"] = subset[list(_REGIONS)].median(axis=1)
    subset = subset.dropna(subset=["producer_price"])
    if (subset["producer_price"] <= 0).any():
        raise ValueError("producer prices must be strictly positive")
    # A semana do produtor comeca na segunda; a da revenda, no domingo anterior.
    subset["date"] = subset["ini"] - pd.Timedelta(days=1)
    frame = subset[["date", "producer_price", "brasil"]].rename(
        columns={"brasil": "producer_price_brasil_raw"}
    ).reset_index(drop=True)
    return frame, _record("anp_producer", ANP_PRODUCER_URL, payload, frame, frame["date"])


def fetch_ipea_series(code: str, name: str) -> tuple[pd.Series, SourceRecord]:
    """Serie diaria do IPEA (Brent em USD/bbl ou dolar comercial)."""

    url = IPEA_URL.format(code=code)
    payload = _get(url)
    values = json.loads(payload)["value"]
    frame = pd.DataFrame(values)
    if frame.empty or not {"VALDATA", "VALVALOR"}.issubset(frame.columns):
        raise ValueError(f"IPEA series {code} came back empty or non-positive")
    frame["date"] = (
        pd.to_datetime(frame["VALDATA"], utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.normalize()
    )
    frame[name] = pd.to_numeric(frame["VALVALOR"], errors="coerce")
    frame = frame.dropna(subset=["date", name]).sort_values("date")
    frame = frame.drop_duplicates("date").reset_index(drop=True)
    if frame.empty or (frame[name] <= 0).any():
        raise ValueError(f"IPEA series {code} came back empty or non-positive")
    return frame.set_index("date")[name], _record(name, url, payload, frame, frame["date"])


def _to_week_index(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Ultimo valor conhecido *antes* do inicio de cada semana da ANP."""

    daily = series.sort_index().resample("D").ffill()
    return daily.reindex(index, method="ffill").to_numpy(float)


def build_causal_panel(
    retail: pd.DataFrame,
    *,
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Combina a revenda com todas as fontes causais e devolve painel e manifesto.

    ``retail`` precisa conter ``date`` (domingo que inicia a semana) e ``price``.
    """

    required = {"date", "price"}
    if not required.issubset(retail.columns):
        raise ValueError(f"retail frame requires columns: {sorted(required)}")
    base = retail[["date", "price"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["price"] = pd.to_numeric(base["price"], errors="coerce")
    base = (
        base.dropna()
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    if len(base) < 120:
        raise ValueError("retail history is too short to build the causal panel")

    records: list[SourceRecord] = []
    ulsd, record = fetch_ulsd(None if raw_dir is None else raw_dir / "yahoo_ulsd.json")
    records.append(record)
    brent, record = fetch_ipea_series("EIA366_PBRENT366", "brent")
    records.append(record)
    usdbrl, record = fetch_ipea_series("GM366_ERC366", "usdbrl")
    records.append(record)
    producer, record = fetch_producer_prices(
        None if raw_dir is None else raw_dir / "anp_produtores_semanal_2013.xls"
    )
    records.append(record)

    index = pd.DatetimeIndex(base["date"])
    base["ulsd"] = _to_week_index(ulsd, index)
    base["brent"] = _to_week_index(brent, index)
    base["usdbrl"] = _to_week_index(usdbrl, index)
    base = base.merge(producer, on="date", how="left")

    missing = base[["ulsd", "brent", "usdbrl"]].isna().any(axis=1)
    if missing.iloc[len(base) // 2 :].any():
        raise ValueError("exogenous series do not cover the recent half of the retail history")

    # Paridade de importacao do diesel em R$/L e custo do cru em R$/bbl.
    base["parity"] = base["ulsd"] / _GALLON_LITERS * base["usdbrl"]
    base["brent_brl"] = base["brent"] * base["usdbrl"]
    base["producer_margin"] = base["price"] - base["producer_price"]

    manifest = {
        "built_at": _now(),
        "n_weeks": int(len(base)),
        "coverage": {
            "start": str(base["date"].min().date()),
            "end": str(base["date"].max().date()),
        },
        "producer_publication_lag_weeks": PRODUCER_PUBLICATION_LAG_WEEKS,
        "producer_aggregate": "median across the five ANP regions",
        "alignment": (
            "daily series forward-filled to the Sunday that starts each ANP week, "
            "so every exogenous value precedes the whole retail measurement window"
        ),
        "sources": [record.as_dict() for record in records],
        "coverage_by_column": {
            column: int(base[column].notna().sum())
            for column in ("ulsd", "brent", "usdbrl", "producer_price", "parity")
        },
    }
    return base, manifest
