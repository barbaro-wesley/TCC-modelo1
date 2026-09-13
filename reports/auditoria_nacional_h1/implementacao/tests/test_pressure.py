"""Pressao de repasse: o atributo so pode usar o que existia na origem."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.pressure import (
    PRESSURE_FEATURES,
    PRESSURE_GATE_Z,
    build_pressure_features,
)


def _causal_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-05", periods=n, freq="7D"),
            "price": 5 + np.cumsum(rng.normal(scale=0.02, size=n)),
            "parity": 3 + np.cumsum(rng.normal(scale=0.02, size=n)),
            "producer_price": 3.5 + np.cumsum(rng.normal(scale=0.02, size=n)),
        }
    )


def test_pressure_never_uses_information_from_the_future() -> None:
    """O teste que importa: mexer no futuro nao pode mexer no passado."""

    frame = _causal_frame()
    baseline = build_pressure_features(frame)

    tampered = frame.copy()
    cut = 150
    tampered.loc[cut:, ["price", "parity", "producer_price"]] *= 3.0
    after = build_pressure_features(tampered)

    # Uma linha usa produtor com tres semanas de defasagem e paridade da origem,
    # entao a contaminacao so pode aparecer a partir de ``cut + 1``.
    pd.testing.assert_series_equal(
        baseline["press_z"].iloc[:cut], after["press_z"].iloc[:cut]
    )
    assert not np.allclose(
        baseline["press_z"].iloc[cut + 4 :].dropna(),
        after["press_z"].iloc[cut + 4 :].dropna(),
    )


def test_standardization_excludes_the_observation_being_standardized() -> None:
    frame = _causal_frame(n=160)

    features = build_pressure_features(frame, warmup=104)

    # Nada padronizado antes do aquecimento; o resto e finito.
    assert features["press_z"].iloc[:104].isna().all()
    assert features["press_z"].iloc[110:].notna().any()


def test_pressure_columns_are_aligned_by_date_and_named_as_declared() -> None:
    frame = _causal_frame()

    features = build_pressure_features(frame)

    assert list(features.columns) == ["date", "press_z", "press1", "dpress1"]
    assert set(PRESSURE_FEATURES) <= set(features.columns)
    pd.testing.assert_series_equal(features["date"], frame["date"])


def test_pressure_scales_the_standardized_gap_by_the_price_level() -> None:
    frame = _causal_frame()

    features = build_pressure_features(frame)

    joined = features.join(frame["price"].shift(1).rename("previous")).dropna()
    ratio = joined["press1"] / joined["press_z"]
    # press1 e press_z multiplicado pelo nivel de preco vigente.
    assert np.allclose(ratio.to_numpy(float), joined["previous"].to_numpy(float))


def test_a_stale_producer_against_a_rising_parity_reads_as_pressure() -> None:
    """O sinal precisa apontar para onde a economia manda."""

    n = 200
    surge = 20
    rng = np.random.default_rng(7)
    calm = 3.0 + rng.normal(scale=0.01, size=n - surge)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-05", periods=n, freq="7D"),
            "price": np.full(n, 5.0),
            "producer_price": np.full(n, 3.5),
            # Paridade sobe forte no fim: o produtor fica barato em relacao a ela.
            "parity": np.concatenate([calm, np.linspace(3.0, 4.5, surge)]),
        }
    )

    features = build_pressure_features(frame)
    pressure = features["press_z"]

    quiet = pressure.iloc[120 : n - surge].dropna()
    surging = pressure.iloc[n - surge + 2 :].dropna()

    # Antes do choque a pressao oscila em torno de zero; durante o choque ela
    # despenca para bem abaixo do portao pre-registrado.
    assert abs(quiet.mean()) < 1.0
    assert surging.max() < PRESSURE_GATE_Z
    assert surging.mean() < quiet.mean() - 2.0


def test_producer_lag_controls_how_stale_the_producer_price_is() -> None:
    frame = _causal_frame()

    near = build_pressure_features(frame, producer_lag=1)
    far = build_pressure_features(frame, producer_lag=6)

    assert not np.allclose(
        near["press_z"].dropna().to_numpy()[:50], far["press_z"].dropna().to_numpy()[:50]
    )


def test_pressure_rejects_incomplete_or_impossible_input() -> None:
    frame = _causal_frame()

    with pytest.raises(ValueError, match="require columns"):
        build_pressure_features(frame.drop(columns=["producer_price"]))
    with pytest.raises(ValueError, match="producer_lag"):
        build_pressure_features(frame, producer_lag=0)
    with pytest.raises(ValueError, match="warmup"):
        build_pressure_features(frame, warmup=4)

    negative = frame.copy()
    negative.loc[3, "parity"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_pressure_features(negative)


def test_missing_producer_weeks_become_missing_pressure_not_wrong_pressure() -> None:
    frame = _causal_frame()
    frame.loc[120:130, "producer_price"] = np.nan

    features = build_pressure_features(frame)

    assert features["press_z"].iloc[123:134].isna().all()
