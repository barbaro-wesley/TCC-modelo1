"""Modelo estadual: alinhamento, causalidade e reversao do spread."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.regional import (
    UF_REGION,
    SpreadConfig,
    SpreadForecaster,
    build_regional_panel,
    normalize_label,
)


def _frames(n: int = 300, seed: int = 0):
    """Estado, nacional e produtor com o mesmo desalinhamento de um dia da ANP."""

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-07", periods=n, freq="7D")
    national = 5 + np.cumsum(rng.normal(scale=0.03, size=n))
    # Spread que reverte para -0,05 com choques pequenos.
    spread = np.empty(n)
    spread[0] = -0.05
    for index in range(1, n):
        spread[index] = -0.05 + 0.9 * (spread[index - 1] + 0.05) + rng.normal(scale=0.01)
    state = pd.DataFrame(
        {"date": dates, "price": national + spread, "stations": 250 + rng.integers(0, 40, n)}
    )
    national_frame = pd.DataFrame({"date": dates, "price": national})
    producer = pd.DataFrame(
        {
            # A semana de competencia do produtor comeca um dia depois.
            "date": dates + pd.Timedelta(days=1),
            "producer_region": 3.5 + np.cumsum(rng.normal(scale=0.02, size=n)),
            "producer_national": 3.6 + np.cumsum(rng.normal(scale=0.02, size=n)),
        }
    )
    return state, national_frame, producer


def test_every_brazilian_state_maps_to_an_anp_region() -> None:
    assert len(UF_REGION) == 27
    assert UF_REGION["RS"] == "sul"
    assert set(UF_REGION.values()) == {"norte", "nordeste", "centro_oeste", "sudeste", "sul"}


def test_labels_are_matched_without_accents_or_case() -> None:
    assert normalize_label(" Rio Grande do Sul ") == "RIO GRANDE DO SUL"
    assert normalize_label("ÓLEO DIESEL S10") == "OLEO DIESEL S10"
    assert normalize_label("Espírito Santo") == "ESPIRITO SANTO"


def test_panel_aligns_the_producer_week_that_starts_one_day_later() -> None:
    """A juncao e por proximidade: alinhar por posicao quebraria com uma lacuna."""

    state, national, producer = _frames()

    panel = build_regional_panel(state, national, producer)

    assert len(panel) == len(state)
    assert panel["producer_spread"].notna().any()
    assert list(panel["date"]) == list(state["date"])


def test_spread_is_the_state_minus_the_national_price() -> None:
    state, national, producer = _frames(n=150)

    panel = build_regional_panel(state, national, producer)

    expected = state["price"].to_numpy(float) - national["price"].to_numpy(float)
    assert np.allclose(panel["spread"].to_numpy(float), expected)
    assert np.allclose(
        panel["spread_lag1"].dropna().to_numpy(float), expected[:-1], atol=1e-12
    )


def test_producer_anchor_is_lagged_and_standardized_causally() -> None:
    state, national, producer = _frames()

    panel = build_regional_panel(state, national, producer, producer_lag=3, warmup=104)

    # Nada padronizado antes do aquecimento.
    assert panel["producer_spread_z"].iloc[:104].isna().all()
    # Mexer no futuro nao pode mexer no passado.
    tampered = producer.copy()
    tampered.loc[200:, ["producer_region", "producer_national"]] *= 2.0
    after = build_regional_panel(state, national, tampered, producer_lag=3, warmup=104)
    pd.testing.assert_series_equal(
        panel["producer_spread_z"].iloc[:200], after["producer_spread_z"].iloc[:200]
    )


def test_panel_rejects_frames_without_the_required_columns() -> None:
    state, national, producer = _frames(n=120)

    with pytest.raises(ValueError, match="state precisa"):
        build_regional_panel(state.drop(columns=["price"]), national, producer)
    with pytest.raises(ValueError, match="producer precisa"):
        build_regional_panel(state, national, producer.drop(columns=["producer_region"]))
    with pytest.raises(ValueError, match="producer_lag"):
        build_regional_panel(state, national, producer, producer_lag=0)


def test_panel_refuses_series_with_no_common_weeks() -> None:
    state, national, producer = _frames(n=120)
    shifted = national.assign(date=national["date"] + pd.Timedelta(days=3))

    with pytest.raises(ValueError, match="nao tem semanas em comum"):
        build_regional_panel(state, shifted, producer)


def test_forecaster_learns_reversion_and_pulls_toward_the_target() -> None:
    state, national, producer = _frames(n=400, seed=3)
    panel = build_regional_panel(state, national, producer)

    model = SpreadForecaster(use_anchor=False).fit(panel)
    summary = model.summary()

    assert summary["fitted"] is True
    assert 0.0 < summary["reversion_kappa"] <= 0.5
    assert summary["half_life_weeks"] is not None

    # Spread muito abaixo do alvo -> previsao sobe; muito acima -> desce.
    low = pd.Series({"spread_lag1": model.mu_ - 0.5, "date": panel["date"].iloc[-1]})
    high = pd.Series({"spread_lag1": model.mu_ + 0.5, "date": panel["date"].iloc[-1]})
    below = model.forecast_row(low, national_point=6.0)
    above = model.forecast_row(high, national_point=6.0)
    assert below.spread_change > 0
    assert above.spread_change < 0
    # No extremo o limite atua, mas limitando a magnitude e preservando o sinal.
    assert below.reason in {"", "variacao_limitada"}
    assert below.spread_change == pytest.approx(-above.spread_change)


def test_state_forecast_is_the_national_point_plus_the_spread() -> None:
    state, national, producer = _frames(n=300, seed=4)
    panel = build_regional_panel(state, national, producer)
    model = SpreadForecaster(use_anchor=False).fit(panel)

    forecast = model.forecast_row(panel.iloc[-1], national_point=6.5)

    assert forecast.state_point == pytest.approx(6.5 + forecast.spread_point)
    assert forecast.national_point == 6.5
    assert "spread_point" in forecast.as_dict()


def test_reversion_is_capped_so_one_odd_window_cannot_flip_the_model() -> None:
    """Meia-vida medida e de semanas, nao de um passo: kappa acima do teto e ruido."""

    n = 300
    rng = np.random.default_rng(5)
    dates = pd.date_range("2018-01-07", periods=n, freq="7D")
    national = np.full(n, 5.0)
    # Spread que alterna de sinal toda semana: kappa bruto proximo de 2.
    spread = 0.1 * ((-1.0) ** np.arange(n)) + rng.normal(scale=0.001, size=n)
    state = pd.DataFrame({"date": dates, "price": national + spread})
    panel = build_regional_panel(
        state,
        pd.DataFrame({"date": dates, "price": national}),
        pd.DataFrame(
            {
                "date": dates + pd.Timedelta(days=1),
                "producer_region": np.full(n, 3.5),
                "producer_national": np.full(n, 3.6),
            }
        ),
    )

    model = SpreadForecaster(use_anchor=False, config=SpreadConfig(max_reversion=0.5)).fit(panel)

    assert model.kappa_ == pytest.approx(0.5)


def test_without_estimable_reversion_the_model_carries_the_spread() -> None:
    n = 300
    dates = pd.date_range("2018-01-07", periods=n, freq="7D")
    rng = np.random.default_rng(6)
    # Passeio aleatorio puro: nao ha reversao a estimar.
    spread = np.cumsum(rng.normal(scale=0.01, size=n))
    national = np.full(n, 5.0)
    panel = build_regional_panel(
        pd.DataFrame({"date": dates, "price": national + spread}),
        pd.DataFrame({"date": dates, "price": national}),
        pd.DataFrame(
            {
                "date": dates + pd.Timedelta(days=1),
                "producer_region": np.full(n, 3.5),
                "producer_national": np.full(n, 3.6),
            }
        ),
    )

    model = SpreadForecaster(use_anchor=False).fit(panel)
    forecast = model.forecast_row(panel.iloc[-1], national_point=5.0)

    assert model.kappa_ >= 0.0
    # Com reversao nula o palpite e o proprio spread de ontem.
    if model.kappa_ == 0.0:
        assert forecast.spread_point == pytest.approx(forecast.origin_spread)


def test_walk_forward_is_causal_and_matches_the_requested_window() -> None:
    state, national, producer = _frames(n=400, seed=7)
    panel = build_regional_panel(state, national, producer)
    start, end = 300, 340
    predictions = panel["national_price"].to_numpy(float)[start:end]

    result = SpreadForecaster(use_anchor=True).walk_forward(panel, predictions, start, end)

    assert len(result) == end - start
    assert list(result["date"]) == list(panel["date"].iloc[start:end])
    assert result["prediction"].notna().all()

    # Estragar o futuro do painel nao muda as previsoes ja emitidas.
    tampered = panel.copy()
    tampered.loc[end:, "spread_lag1"] *= 10.0
    again = SpreadForecaster(use_anchor=True).walk_forward(tampered, predictions, start, end)
    assert np.allclose(result["prediction"], again["prediction"])


def test_walk_forward_validates_its_inputs() -> None:
    state, national, producer = _frames(n=300)
    panel = build_regional_panel(state, national, producer)
    model = SpreadForecaster()

    with pytest.raises(ValueError, match="janela de walk-forward invalida"):
        model.walk_forward(panel, np.zeros(10), 0, 10)
    with pytest.raises(ValueError, match="nao cobrem a janela"):
        model.walk_forward(panel, np.zeros(5), 200, 240)


def test_forecast_rejects_an_impossible_national_point() -> None:
    state, national, producer = _frames(n=300)
    panel = build_regional_panel(state, national, producer)
    model = SpreadForecaster(use_anchor=False).fit(panel)

    with pytest.raises(ValueError, match="national_point"):
        model.forecast_row(panel.iloc[-1], national_point=0.0)


def test_missing_spread_falls_back_instead_of_guessing() -> None:
    state, national, producer = _frames(n=300)
    panel = build_regional_panel(state, national, producer)
    model = SpreadForecaster(use_anchor=False).fit(panel)

    forecast = model.forecast_row(pd.Series({"date": panel["date"].iloc[-1]}), national_point=6.0)

    assert forecast.fallback_used is True
    assert forecast.reason == "spread_indisponivel"


def test_fit_refuses_a_history_shorter_than_the_declared_minimum() -> None:
    state, national, producer = _frames(n=140)
    panel = build_regional_panel(state, national, producer)

    with pytest.raises(ValueError, match="historico insuficiente"):
        SpreadForecaster(config=SpreadConfig(min_train=200), use_anchor=False).fit(panel)
    with pytest.raises(ValueError, match="end esta fora do painel"):
        SpreadForecaster(use_anchor=False).fit(panel, end=0)


def test_summary_is_explicit_before_being_fitted() -> None:
    assert SpreadForecaster().summary() == {"fitted": False}


def test_the_change_limit_caps_the_magnitude_and_keeps_the_direction() -> None:
    """Zerar a reversao no extremo silenciaria o modelo onde ele mais importa."""

    state, national, producer = _frames(n=400, seed=8)
    panel = build_regional_panel(state, national, producer)
    model = SpreadForecaster(
        use_anchor=False, config=SpreadConfig(change_limit_sigma=0.5)
    ).fit(panel)

    extreme = pd.Series({"spread_lag1": model.mu_ - 5.0, "date": panel["date"].iloc[-1]})
    forecast = model.forecast_row(extreme, national_point=6.0)

    limit = 0.5 * model.sigma_
    assert forecast.spread_change == pytest.approx(limit)
    assert forecast.fallback_used is True
    assert forecast.reason == "variacao_limitada"


def _states_workbook(rows: list[dict[str, object]]) -> bytes:
    """Reproduz o formato real da planilha estadual: 17 linhas de nota, depois a tabela."""

    import io as _io

    frame = pd.DataFrame(rows)
    buffer = _io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"nota": ["AGENCIA NACIONAL DO PETROLEO"] * 17}).to_excel(
            writer, sheet_name="ESTADOS", index=False, header=False
        )
        frame.to_excel(writer, sheet_name="ESTADOS", index=False, startrow=17)
    return buffer.getvalue()


def _state_rows(n: int = 6, product: str = "ÓLEO DIESEL S10", unit: str = "R$/l"):
    dates = pd.date_range("2024-01-07", periods=n, freq="7D")
    rows = []
    for index, date in enumerate(dates):
        for estado, preco in (("RIO GRANDE DO SUL", 6.0 + index * 0.01), ("SÃO PAULO", 7.0)):
            rows.append(
                {
                    "DATA INICIAL": date,
                    "DATA FINAL": date + pd.Timedelta(days=6),
                    "REGIÃO": "SUL" if estado.startswith("RIO") else "SUDESTE",
                    "ESTADO": estado,
                    "PRODUTO": product,
                    "NÚMERO DE POSTOS PESQUISADOS": 250 + index,
                    "UNIDADE DE MEDIDA": unit,
                    "PREÇO MÉDIO REVENDA": preco,
                }
            )
    return rows


def test_state_fetch_selects_one_state_one_product_and_records_provenance(monkeypatch, tmp_path):
    from vs_epl_krls import regional

    payload = _states_workbook(_state_rows())
    monkeypatch.setattr(regional, "_get", lambda url: payload)
    cache = tmp_path / "estados.xlsx"

    frame, record = regional.fetch_state_weekly("rs", cache=cache)

    assert len(frame) == 6
    assert list(frame.columns) == ["date", "price", "stations"]
    assert frame["price"].iloc[0] == pytest.approx(6.0)
    assert record.name == "anp_estado_rs"
    assert record.rows == 6
    assert record.coverage_start == "2024-01-07"
    assert len(record.sha256) == 64
    # O cache guarda o arquivo bruto, para auditoria posterior.
    assert cache.read_bytes() == payload


def test_state_fetch_refuses_an_unknown_uf(monkeypatch):
    from vs_epl_krls import regional

    monkeypatch.setattr(regional, "_get", lambda url: b"")
    with pytest.raises(ValueError, match="unidade da federacao desconhecida"):
        regional.fetch_state_weekly("XX")


def test_state_fetch_fails_loudly_when_the_anp_changes_the_layout(monkeypatch):
    """Formato novo tem de quebrar alto: coluna faltando ja custou uma serie inteira aqui."""

    from vs_epl_krls import regional

    rows = _state_rows()
    for row in rows:
        del row["UNIDADE DE MEDIDA"]
    monkeypatch.setattr(regional, "_get", lambda url: _states_workbook(rows))

    with pytest.raises(ValueError, match="mudou de formato"):
        regional.fetch_state_weekly("RS")


def test_state_fetch_rejects_an_unexpected_unit(monkeypatch):
    from vs_epl_krls import regional

    monkeypatch.setattr(
        regional, "_get", lambda url: _states_workbook(_state_rows(unit="R$/m3"))
    )

    with pytest.raises(ValueError, match="unidade inesperada"):
        regional.fetch_state_weekly("RS")


def test_state_fetch_reports_when_the_product_is_absent(monkeypatch):
    from vs_epl_krls import regional

    monkeypatch.setattr(
        regional, "_get", lambda url: _states_workbook(_state_rows(product="GASOLINA COMUM"))
    )

    with pytest.raises(ValueError, match="nenhuma linha de Diesel S10"):
        regional.fetch_state_weekly("RS")


def _producer_workbook(n: int = 8) -> bytes:
    """Formato do arquivo de produtores: nove linhas de cabecalho, sem nomes de coluna."""

    import io as _io

    from vs_epl_krls.causal_ingest import PRODUCER_PRODUCT

    dates = pd.date_range("2024-01-08", periods=n, freq="7D")
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            [
                PRODUCER_PRODUCT,
                date,
                date + pd.Timedelta(days=6),
                3.90,  # norte
                3.95,  # nordeste
                3.80,  # centro_oeste
                3.60 + index * 0.01,  # sul
                3.70,  # sudeste
                9.99,  # brasil, deliberadamente defeituoso
            ]
        )
        rows.append(["GASOLINA A", date, date + pd.Timedelta(days=6), 1, 1, 1, 1, 1, 1])
    buffer = _io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame([["cabecalho"]] * 9).to_excel(
            writer, sheet_name="p", index=False, header=False
        )
        pd.DataFrame(rows).to_excel(writer, sheet_name="p", index=False, header=False, startrow=9)
    return buffer.getvalue()


def test_regional_producer_uses_the_region_and_a_robust_national_aggregate(monkeypatch):
    """A coluna Brasil do arquivo oficial e defeituosa; a mediana entre regioes nao."""

    from vs_epl_krls import regional

    monkeypatch.setattr(regional, "_get", lambda url: _producer_workbook())

    frame, record = regional.fetch_regional_producer("sul")

    assert list(frame.columns) == ["date", "producer_region", "producer_national"]
    assert frame["producer_region"].iloc[0] == pytest.approx(3.60)
    # Mediana de (3.90, 3.95, 3.80, 3.60, 3.70) = 3.80, e nao os 9.99 da coluna Brasil.
    assert frame["producer_national"].iloc[0] == pytest.approx(3.80)
    assert record.name == "anp_producer_sul"


def test_regional_producer_refuses_an_unknown_region(monkeypatch):
    from vs_epl_krls import regional

    monkeypatch.setattr(regional, "_get", lambda url: b"")
    with pytest.raises(ValueError, match="regiao desconhecida"):
        regional.fetch_regional_producer("pampa")


def test_walk_forward_marks_weeks_it_cannot_predict() -> None:
    """Sem previsao nacional finita nao ha previsao estadual: a linha sai como fallback."""

    state, national, producer = _frames(n=300, seed=9)
    panel = build_regional_panel(state, national, producer)
    start, end = 250, 260
    predictions = panel["national_price"].to_numpy(float)[start:end].copy()
    predictions[3] = np.nan

    result = SpreadForecaster(use_anchor=False).walk_forward(panel, predictions, start, end)

    assert bool(result["fallback"].iloc[3]) is True
    assert np.isnan(result["prediction"].iloc[3])
    assert result["prediction"].drop(index=3).notna().all()


def test_walk_forward_holds_back_until_it_can_be_fitted() -> None:
    state, national, producer = _frames(n=300, seed=10)
    panel = build_regional_panel(state, national, producer)
    # Janela que comeca antes do minimo de treino declarado.
    start, end = 10, 30
    predictions = panel["national_price"].to_numpy(float)[start:end]

    result = SpreadForecaster(use_anchor=False).walk_forward(panel, predictions, start, end)

    assert result["prediction"].isna().all()
    assert result["fallback"].all()


def test_walk_forward_carries_the_pooling_into_the_internal_model() -> None:
    """Sem propagar, a avaliacao mediria sempre o ajuste local e o pooling seria invisivel."""

    state, national, producer = _frames(n=400, seed=11)
    panel = build_regional_panel(state, national, producer)
    start, end = 300, 340
    predictions = panel["national_price"].to_numpy(float)[start:end]

    local = SpreadForecaster(use_anchor=False).walk_forward(panel, predictions, start, end)
    pooled = SpreadForecaster(
        use_anchor=False, pooled_kappa=0.45, pooling_weight=0.0
    ).walk_forward(panel, predictions, start, end)

    # Peso proprio zero: a reversao usada tem de ser a do conjunto, e a previsao muda.
    assert not np.allclose(local["prediction"], pooled["prediction"])


def test_pooling_shrinks_noisy_states_toward_the_group() -> None:
    """Peso de Bayes empirico: quem mede mal toma emprestado, quem mede bem não."""

    from vs_epl_krls.regional import pool_reversion

    estimates = {
        "SP": (0.10, 0.005),   # estimativa precisa
        "MG": (0.12, 0.005),
        "PR": (0.11, 0.005),
        "RO": (0.40, 0.200),   # estimativa muito ruidosa e distante
    }

    pool = pool_reversion(estimates)

    assert pool.n_states == 4
    assert pool.weights["SP"] > pool.weights["RO"]
    # O estado ruidoso é puxado com força para o conjunto; o preciso quase não se move.
    assert abs(pool.shrunk_kappa["RO"] - 0.40) > abs(pool.shrunk_kappa["SP"] - 0.10)
    assert 0.09 < pool.pooled_kappa < 0.20


def test_pooling_collapses_when_the_states_do_not_really_differ() -> None:
    """Sem heterogeneidade real, tau² vai a zero e todos adotam o valor comum."""

    from vs_epl_krls.regional import pool_reversion

    # Estimativas idênticas com erro grande: a variação observada cabe no erro.
    pool = pool_reversion({uf: (0.10, 0.20) for uf in ("SP", "MG", "PR", "RS", "GO")})

    assert pool.between_variance == pytest.approx(0.0)
    assert all(weight == pytest.approx(0.0) for weight in pool.weights.values())
    assert all(
        value == pytest.approx(pool.pooled_kappa) for value in pool.shrunk_kappa.values()
    )


def test_pooling_keeps_states_apart_when_they_genuinely_differ() -> None:
    from vs_epl_krls.regional import pool_reversion

    # Estimativas precisas e muito diferentes: nada a tomar emprestado.
    pool = pool_reversion(
        {"A": (0.05, 0.001), "B": (0.20, 0.001), "C": (0.35, 0.001), "D": (0.10, 0.001)}
    )

    assert pool.between_variance > 0
    assert all(weight > 0.99 for weight in pool.weights.values())


def test_pooling_declines_without_enough_states(caplog) -> None:
    from vs_epl_krls.regional import pool_reversion

    pool = pool_reversion({"SP": (0.1, 0.01), "MG": (0.2, 0.01)})

    assert pool.n_states == 2
    assert pool.weights == {"SP": 1.0, "MG": 1.0}
    assert pool.shrunk_kappa["SP"] == pytest.approx(0.1)


def test_pooling_discards_estimates_without_a_usable_standard_error() -> None:
    from vs_epl_krls.regional import pool_reversion

    pool = pool_reversion(
        {
            "SP": (0.10, 0.01),
            "MG": (0.12, 0.01),
            "PR": (0.11, 0.01),
            "XX": (0.50, 0.0),            # erro-padrão zero
            "YY": (float("nan"), 0.01),   # estimativa inválida
        }
    )

    assert pool.n_states == 3
    assert set(pool.weights) == {"SP", "MG", "PR"}


def test_station_weighting_downweights_the_weeks_with_a_thin_sample() -> None:
    """Semana de 8 postos não pode pesar como semana de 200."""

    state, national, producer = _frames(n=400, seed=12)
    panel = build_regional_panel(state, national, producer)
    # Contamina as semanas de amostra fina com um choque que não é preço real.
    thin = panel.index[panel.index % 7 == 0]
    panel.loc[thin, "stations"] = 8
    panel.loc[thin, "y"] = panel.loc[thin, "y"] + 0.5

    plain = SpreadForecaster(use_anchor=False, weight_by_stations=False).fit(panel)
    weighted = SpreadForecaster(use_anchor=False, weight_by_stations=True).fit(panel)

    assert weighted.summary()["weighted_by_stations"] is True
    assert plain.summary()["weighted_by_stations"] is False
    # A ponderação muda o ajuste — é o ponto.
    assert weighted.mu_ != pytest.approx(plain.mu_)


def test_station_weighting_is_inert_without_the_column() -> None:
    state, national, producer = _frames(n=300, seed=13)
    panel = build_regional_panel(state.drop(columns=["stations"]), national, producer)

    plain = SpreadForecaster(use_anchor=False).fit(panel)
    weighted = SpreadForecaster(use_anchor=False, weight_by_stations=True).fit(panel)

    assert weighted.kappa_ == pytest.approx(plain.kappa_)
