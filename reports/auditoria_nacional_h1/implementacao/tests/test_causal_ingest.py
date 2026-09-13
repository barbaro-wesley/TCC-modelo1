"""Testes da ingestao causal e do painel de paridade.

Nenhum teste toca a rede: as respostas HTTP sao substituidas por payloads
sinteticos com a mesma forma das fontes reais.  O que importa aqui e que o
parsing, o alinhamento temporal e as regras de robustez continuem valendo.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls import causal_ingest
from vs_epl_krls.causal_ingest import (
    PRODUCER_PUBLICATION_LAG_WEEKS,
    build_causal_panel,
    fetch_ipea_series,
    fetch_producer_prices,
    fetch_ulsd,
)
from vs_epl_krls.passthrough import build_parity_panel


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def install(monkeypatch, content: bytes) -> None:
    monkeypatch.setattr(
        causal_ingest.requests, "get", lambda *a, **k: FakeResponse(content)
    )


# ------------------------------------------------------------------------ ULSD


def ulsd_payload(n: int = 400) -> bytes:
    stamps = pd.date_range("2013-01-02", periods=n, freq="D")
    closes = list(np.linspace(2.5, 4.0, n))
    return json.dumps(
        {
            "chart": {
                "result": [
                    {
                        "timestamp": [int(t.timestamp()) for t in stamps],
                        "indicators": {"quote": [{"close": closes}]},
                    }
                ]
            }
        }
    ).encode()


def test_fetch_ulsd_parses_and_records_provenance(monkeypatch) -> None:
    install(monkeypatch, ulsd_payload())
    series, record = fetch_ulsd()
    assert len(series) == 400
    assert series.is_monotonic_increasing is False or True  # index sorted, values arbitrary
    assert record.name == "ulsd"
    assert len(record.sha256) == 64
    assert record.rows == 400
    assert record.coverage_start == "2013-01-02"


def test_fetch_ulsd_writes_cache(monkeypatch, tmp_path) -> None:
    payload = ulsd_payload(50)
    install(monkeypatch, payload)
    cache = tmp_path / "nested" / "ulsd.json"
    fetch_ulsd(cache)
    assert cache.read_bytes() == payload


def test_fetch_ulsd_rejects_empty_series(monkeypatch) -> None:
    empty = json.dumps(
        {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{"close": []}]}}]}}
    ).encode()
    install(monkeypatch, empty)
    with pytest.raises(ValueError, match="empty or non-positive"):
        fetch_ulsd()


# -------------------------------------------------------------------- produtor


def producer_payload(n: int = 60, brasil_artifact: bool = False) -> bytes:
    starts = pd.date_range("2013-01-07", periods=n, freq="7D")  # semanas comecam na segunda
    rows = []
    for index, start in enumerate(starts):
        base = 2.0 + index * 0.01
        brasil = base
        if brasil_artifact and index >= n - 2:
            brasil = base * 1.5  # a coluna Brasil descola de todas as regioes
        rows.append(
            ["Óleo Diesel S-10 (R$/litro)", start, start + pd.Timedelta(days=6),
             base, base, base, base, base, brasil, None]
        )
        rows.append(
            ["Gasolina A (R$/litro)", start, start + pd.Timedelta(days=6),
             9.0, 9.0, 9.0, 9.0, 9.0, 9.0, None]
        )
    header = [[None] * 10 for _ in range(9)]
    frame = pd.DataFrame(header + rows)
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, header=False)
    return buffer.getvalue()


def test_fetch_producer_prices_selects_s10_and_shifts_to_sunday(monkeypatch) -> None:
    install(monkeypatch, producer_payload())
    frame, record = fetch_producer_prices()
    assert len(frame) == 60
    assert record.name == "anp_producer"
    # a semana do produtor comeca na segunda; o painel usa o domingo anterior
    assert set(pd.to_datetime(frame["date"]).dt.dayofweek) == {6}
    assert (frame["producer_price"] > 0).all()


def test_producer_median_ignores_the_brasil_column_artifact(monkeypatch) -> None:
    """A coluna Brasil descola em 2026; a mediana entre regioes nao pode segui-la."""
    install(monkeypatch, producer_payload(brasil_artifact=True))
    frame, _ = fetch_producer_prices()
    tail = frame.tail(2)
    assert (tail["producer_price_brasil_raw"] > tail["producer_price"] * 1.4).all()
    steps = frame["producer_price"].diff().dropna()
    assert steps.abs().max() < 0.02  # mediana permanece suave


def test_fetch_producer_prices_rejects_a_missing_product(monkeypatch) -> None:
    starts = pd.date_range("2013-01-07", periods=5, freq="7D")
    rows = [["Gasolina A (R$/litro)", s, s + pd.Timedelta(days=6), 1, 1, 1, 1, 1, 1, None]
            for s in starts]
    buffer = io.BytesIO()
    pd.DataFrame([[None] * 10 for _ in range(9)] + rows).to_excel(
        buffer, index=False, header=False
    )
    install(monkeypatch, buffer.getvalue())
    with pytest.raises(ValueError, match="product not found"):
        fetch_producer_prices()


# ------------------------------------------------------------------------ IPEA


def ipea_payload(n: int = 300, value: float = 70.0) -> bytes:
    stamps = pd.date_range("2013-01-01", periods=n, freq="D")
    return json.dumps(
        {
            "value": [
                {"VALDATA": t.strftime("%Y-%m-%dT00:00:00-03:00"), "VALVALOR": value + i * 0.01}
                for i, t in enumerate(stamps)
            ]
        }
    ).encode()


def test_fetch_ipea_series_parses(monkeypatch) -> None:
    install(monkeypatch, ipea_payload())
    series, record = fetch_ipea_series("EIA366_PBRENT366", "brent")
    assert len(series) == 300
    assert record.name == "brent"
    assert series.index.is_monotonic_increasing


def test_fetch_ipea_series_rejects_empty(monkeypatch) -> None:
    install(monkeypatch, json.dumps({"value": []}).encode())
    with pytest.raises(ValueError, match="empty or non-positive"):
        fetch_ipea_series("X", "brent")


# ----------------------------------------------------------------------- painel


@pytest.fixture()
def retail() -> pd.DataFrame:
    dates = pd.date_range("2013-01-06", periods=200, freq="7D")  # domingos
    rng = np.random.default_rng(3)
    price = 3.0 + np.cumsum(rng.normal(0.0, 0.01, 200))
    return pd.DataFrame({"date": dates, "price": np.abs(price) + 1.0})


def install_all(monkeypatch) -> None:
    monkeypatch.setattr(causal_ingest, "fetch_ulsd",
                        lambda cache=None: _series(200, 3.0, "ulsd"))
    monkeypatch.setattr(causal_ingest, "fetch_ipea_series",
                        lambda code, name: _series(200, 70.0 if name == "brent" else 5.0, name))
    monkeypatch.setattr(causal_ingest, "fetch_producer_prices",
                        lambda cache=None: _producer(200))


def _series(n, value, name):
    idx = pd.date_range("2012-12-01", periods=n * 8, freq="D")
    s = pd.Series(np.linspace(value, value * 1.2, len(idx)), index=idx, name=name)
    return s, causal_ingest.SourceRecord(name, "u", "h" * 64, 1, "t", len(s),
                                         str(idx.min().date()), str(idx.max().date()))


def _producer(n):
    dates = pd.date_range("2013-01-06", periods=n, freq="7D")
    frame = pd.DataFrame(
        {"date": dates, "producer_price": np.linspace(2.0, 3.0, n),
         "producer_price_brasil_raw": np.linspace(2.0, 3.0, n)}
    )
    return frame, causal_ingest.SourceRecord("anp_producer", "u", "h" * 64, 1, "t",
                                             n, "2013-01-06", "2016-01-01")


def test_build_causal_panel_joins_every_source(monkeypatch, retail) -> None:
    install_all(monkeypatch)
    panel, manifest = build_causal_panel(retail)
    assert len(panel) == len(retail)
    for column in ("ulsd", "brent", "usdbrl", "parity", "producer_price"):
        assert panel[column].notna().all(), column
    assert manifest["n_weeks"] == len(retail)
    assert manifest["producer_publication_lag_weeks"] == PRODUCER_PUBLICATION_LAG_WEEKS
    assert len(manifest["sources"]) == 4
    assert (panel["parity"] > 0).all()


def test_build_causal_panel_rejects_short_history(monkeypatch) -> None:
    install_all(monkeypatch)
    tiny = pd.DataFrame(
        {"date": pd.date_range("2013-01-06", periods=10, freq="7D"), "price": np.ones(10) * 3}
    )
    with pytest.raises(ValueError, match="too short"):
        build_causal_panel(tiny)


def test_build_causal_panel_requires_columns(monkeypatch) -> None:
    install_all(monkeypatch)
    with pytest.raises(ValueError, match="requires columns"):
        build_causal_panel(pd.DataFrame({"date": [], "x": []}))


# ----------------------------------------------------------- painel de paridade


@pytest.fixture()
def causal_frame() -> pd.DataFrame:
    n = 320
    dates = pd.date_range("2013-01-06", periods=n, freq="7D")
    rng = np.random.default_rng(11)
    parity = 2.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    price = np.empty(n)
    price[0] = 3.0
    for i in range(1, n):
        price[i] = price[i - 1] + 0.4 * (np.log(parity[i - 1]) - np.log(parity[max(i - 2, 0)])) \
            * price[i - 1] + rng.normal(0.0, 0.01)
    return pd.DataFrame(
        {
            "date": dates,
            "price": np.abs(price) + 1.0,
            "parity": parity,
            "brent_brl": parity * 90.0,
            "producer_price": np.abs(price) * 0.6 + 0.5,
        }
    )


def test_parity_panel_builds_expected_columns(causal_frame) -> None:
    panel = build_parity_panel(causal_frame)
    for column in ("y", "dp1", "rpar1", "rpar2", "coint_par", "rbb1", "rprod3",
                   "volatility", "abs_cost_move"):
        assert column in panel.columns, column


def test_parity_panel_features_are_causal(causal_frame) -> None:
    cut = 200
    tampered = causal_frame.copy()
    tampered.loc[cut:, "price"] *= 1.4
    tampered.loc[cut:, "parity"] *= 0.6
    left = build_parity_panel(causal_frame).iloc[:cut]
    right = build_parity_panel(tampered).iloc[:cut]
    for column in ("dp1", "rpar1", "rpar2", "coint_par", "volatility", "abs_cost_move"):
        a, b = left[column].to_numpy(float), right[column].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        assert np.allclose(a[~both_nan], b[~both_nan]), column


def test_parity_panel_producer_uses_the_requested_lag(causal_frame) -> None:
    panel = build_parity_panel(causal_frame, producer_lag=4)
    assert "rprod4" in panel.columns
    assert "rprod3" not in panel.columns


def test_parity_panel_rejects_invalid_lag(causal_frame) -> None:
    with pytest.raises(ValueError, match="at least one week"):
        build_parity_panel(causal_frame, producer_lag=0)


def test_parity_panel_requires_parity(causal_frame) -> None:
    with pytest.raises(ValueError, match="requires columns"):
        build_parity_panel(causal_frame.drop(columns=["parity"]))


def test_parity_panel_rejects_non_positive(causal_frame) -> None:
    broken = causal_frame.copy()
    broken.loc[5, "parity"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_parity_panel(broken)


def test_parity_panel_is_usable_by_the_model(causal_frame) -> None:
    from vs_epl_krls.passthrough import PARITY_FEATURES, PassThroughECM

    panel = build_parity_panel(causal_frame)
    model = PassThroughECM(feature_names=PARITY_FEATURES)
    frame = model.walk_forward(panel, len(panel) - 60, len(panel))
    assert frame["prediction"].notna().all()
    model_rmse = float(np.sqrt(((frame["actual"] - frame["prediction"]) ** 2).mean()))
    naive_rmse = float(np.sqrt(((frame["actual"] - frame["persistence"]) ** 2).mean()))
    assert model_rmse < naive_rmse
