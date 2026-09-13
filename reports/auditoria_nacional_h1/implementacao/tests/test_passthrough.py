"""Testes do modelo de repasse e do intervalo condicional.

O teste mais importante aqui e o de causalidade: nenhum atributo da linha ``T``
pode mudar quando o futuro muda.  Sem essa garantia, todo o resto do protocolo
temporal do repositorio perde sentido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.conditional_interval import ConditionalIntervalCalibrator
from vs_epl_krls.passthrough import (
    PassThroughConfig,
    PassThroughECM,
    _huber_irls,
    build_passthrough_panel,
)


def synthetic_frame(n: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-06", periods=n, freq="7D")
    brent = 60.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    usdbrl = 3.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    cost = np.log(brent * usdbrl)
    price = np.empty(n)
    price[0] = 3.0
    for index in range(1, n):
        pass_through = 0.35 * (cost[index - 1] - cost[max(index - 2, 0)]) * price[index - 1]
        price[index] = price[index - 1] + pass_through + rng.normal(0.0, 0.01)
    return pd.DataFrame(
        {"date": dates, "price": np.abs(price) + 0.5, "brent": brent, "usdbrl": usdbrl}
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return synthetic_frame()


@pytest.fixture(scope="module")
def panel(frame: pd.DataFrame) -> pd.DataFrame:
    return build_passthrough_panel(frame)


# --------------------------------------------------------------------- painel


def test_panel_has_expected_columns(panel: pd.DataFrame) -> None:
    expected = {
        "date", "price", "origin_price", "y", "dp1", "rbb1", "rbb2",
        "coint_r", "volatility", "abs_cost_move",
    }
    assert expected.issubset(panel.columns)
    assert len(panel) > 0


def test_panel_target_is_the_weekly_change(panel: pd.DataFrame) -> None:
    reconstructed = panel["origin_price"] + panel["y"]
    valid = reconstructed.notna()
    assert np.allclose(reconstructed[valid], panel["price"][valid])


def test_features_are_causal(frame: pd.DataFrame) -> None:
    """Alterar o futuro nao pode mexer em nenhum atributo do passado."""
    cut = 200
    tampered = frame.copy()
    tampered.loc[cut:, "price"] *= 1.5
    tampered.loc[cut:, "brent"] *= 2.0
    tampered.loc[cut:, "usdbrl"] *= 0.5

    original = build_passthrough_panel(frame).iloc[:cut]
    modified = build_passthrough_panel(tampered).iloc[:cut]
    for column in ("dp1", "rbb1", "rbb2", "coint_r", "volatility", "abs_cost_move"):
        left = original[column].to_numpy(float)
        right = modified[column].to_numpy(float)
        both_nan = np.isnan(left) & np.isnan(right)
        assert np.allclose(left[~both_nan], right[~both_nan], equal_nan=True), column


def test_panel_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="requires columns"):
        build_passthrough_panel(pd.DataFrame({"date": [], "price": []}))


def test_panel_rejects_short_history(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="longer history"):
        build_passthrough_panel(frame.head(30))


def test_panel_rejects_non_positive_price(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[10, "price"] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_passthrough_panel(broken)


def test_panel_rejects_non_positive_cost(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[10, "brent"] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        build_passthrough_panel(broken)


# ---------------------------------------------------------------- configuracao


@pytest.mark.parametrize(
    "kwargs",
    [
        {"huber_delta": 0.0},
        {"huber_delta": float("nan")},
        {"irls_iterations": 0},
        {"min_train": 5},
        {"interval_nominal": 1.0},
        {"interval_nominal": 0.1},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PassThroughConfig(**kwargs)


# ------------------------------------------------------------------- regressao


def test_huber_irls_resists_a_single_outlier() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 2.0 * x + rng.normal(0.0, 0.05, 200)
    design = np.column_stack([np.ones(200), x])
    clean = _huber_irls(design, y, delta=1.345, iterations=8)
    contaminated = y.copy()
    contaminated[0] += 60.0
    robust = _huber_irls(design, contaminated, delta=1.345, iterations=8)
    ols, *_ = np.linalg.lstsq(design, contaminated, rcond=None)
    assert clean is not None and robust is not None
    assert abs(robust[1] - 2.0) < abs(ols[1] - 2.0)


def test_huber_irls_returns_finite_or_none() -> None:
    design = np.ones((80, 2))
    target = np.arange(80, dtype=float)
    result = _huber_irls(design, target, delta=1.345, iterations=4)
    assert result is None or np.all(np.isfinite(result))


# ----------------------------------------------------------------------- ajuste


def test_fit_recovers_the_pass_through_sign(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel)
    assert model.coefficients_ is not None
    summary = model.summary()
    assert summary["fitted"] is True
    assert summary["n_train"] > 0
    assert summary["coefficients"]["rbb1"] > 0.0


def test_fit_only_uses_rows_before_end(panel: pd.DataFrame) -> None:
    tampered = panel.copy()
    tampered.loc[tampered.index >= 200, "y"] = 99.0
    left = PassThroughECM().fit(panel, end=200).coefficients_
    right = PassThroughECM().fit(tampered, end=200).coefficients_
    assert np.allclose(left, right)


def test_fit_rejects_out_of_range_end(panel: pd.DataFrame) -> None:
    model = PassThroughECM()
    with pytest.raises(ValueError, match="outside the panel"):
        model.fit(panel, end=len(panel) + 1)
    with pytest.raises(ValueError, match="outside the panel"):
        model.fit(panel, end=0)


def test_fit_rejects_panel_without_target(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="target column"):
        PassThroughECM().fit(panel.drop(columns=["y"]))


def test_fit_rejects_insufficient_rows(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not enough usable rows"):
        PassThroughECM().fit(panel, end=61)


def test_predict_requires_a_fitted_model(panel: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        PassThroughECM().predict_delta(panel.iloc[-1])


# --------------------------------------------------------------------- previsao


def test_forecast_row_produces_an_ordered_interval(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel, end=len(panel) - 1)
    row = panel.iloc[-1]
    forecast = model.forecast_row(row, origin_price=float(row["origin_price"]))
    assert forecast.lower < forecast.point < forecast.upper
    assert forecast.conditional_sigma > 0
    assert np.isfinite(forecast.delta)
    payload = forecast.as_dict()
    assert payload["nominal_coverage"] == pytest.approx(0.80)
    assert payload["fallback_used"] is False


def test_forecast_falls_back_when_features_are_missing(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel, end=len(panel) - 1)
    row = panel.iloc[-1].copy()
    row["rbb1"] = np.nan
    forecast = model.forecast_row(row, origin_price=float(panel["origin_price"].iloc[-1]))
    assert forecast.fallback_used is True
    assert forecast.reason == "features_unavailable"
    assert forecast.point == pytest.approx(float(panel["origin_price"].iloc[-1]))


def test_forecast_falls_back_on_an_implausible_change(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel, end=len(panel) - 1)
    row = panel.iloc[-1].copy()
    row["rbb1"] = 5_000.0
    forecast = model.forecast_row(row, origin_price=float(row["origin_price"]))
    assert forecast.fallback_used is True
    assert forecast.reason == "implausible_change"


def test_forecast_rejects_invalid_origin_price(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel, end=len(panel) - 1)
    with pytest.raises(ValueError, match="finite and positive"):
        model.forecast_row(panel.iloc[-1], origin_price=0.0)


def test_predict_delta_returns_nan_on_missing_features(panel: pd.DataFrame) -> None:
    model = PassThroughECM().fit(panel, end=len(panel) - 1)
    row = panel.iloc[-1].copy()
    row["coint_r"] = np.nan
    assert np.isnan(model.predict_delta(row))


# ------------------------------------------------------------------ walkforward


def test_walk_forward_is_prequential_and_finite(panel: pd.DataFrame) -> None:
    start = len(panel) - 40
    frame = PassThroughECM().walk_forward(panel, start, len(panel))
    assert len(frame) == 40
    assert frame["prediction"].notna().all()
    assert np.isfinite(frame["prediction"]).all()
    assert (frame["lower"] <= frame["upper"]).all()


def test_walk_forward_ignores_the_future(panel: pd.DataFrame) -> None:
    start, end = len(panel) - 40, len(panel) - 20
    tampered = panel.copy()
    tampered.loc[tampered.index >= end, "y"] = 42.0
    tampered.loc[tampered.index >= end, "price"] = 42.0
    left = PassThroughECM().walk_forward(panel, start, end)["prediction"].to_numpy()
    right = PassThroughECM().walk_forward(tampered, start, end)["prediction"].to_numpy()
    assert np.allclose(left, right)


def test_walk_forward_rejects_an_invalid_window(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="invalid walk-forward window"):
        PassThroughECM().walk_forward(panel, 0, 10)
    with pytest.raises(ValueError, match="invalid walk-forward window"):
        PassThroughECM().walk_forward(panel, 10, len(panel) + 5)


def test_walk_forward_beats_persistence_on_the_synthetic_stream(panel: pd.DataFrame) -> None:
    """O fluxo sintetico tem repasse real, entao o modelo deve captura-lo."""
    start = len(panel) - 80
    frame = PassThroughECM().walk_forward(panel, start, len(panel))
    model_rmse = float(np.sqrt(((frame["actual"] - frame["prediction"]) ** 2).mean()))
    naive_rmse = float(np.sqrt(((frame["actual"] - frame["persistence"]) ** 2).mean()))
    assert model_rmse < naive_rmse


def test_summary_before_fit_reports_unfitted() -> None:
    assert PassThroughECM().summary() == {"fitted": False}


# ------------------------------------------------------------------- intervalo


def test_calibrator_produces_ordered_bands(panel: pd.DataFrame) -> None:
    residuals = np.random.default_rng(3).normal(0.0, 0.02, len(panel))
    calibrator = ConditionalIntervalCalibrator().fit(residuals, panel)
    band = calibrator.band(6.5, panel.iloc[-1])
    assert band.lower < 6.5 < band.upper
    assert band.width > 0
    assert calibrator.summary()["fitted"] is True


def test_calibrator_widens_when_volatility_rises(panel: pd.DataFrame) -> None:
    rng = np.random.default_rng(11)
    scale = panel["volatility"].fillna(panel["volatility"].median()).to_numpy(float)
    residuals = rng.normal(0.0, 1.0, len(panel)) * np.nan_to_num(scale, nan=0.01)
    calibrator = ConditionalIntervalCalibrator().fit(residuals, panel)
    calm = panel.iloc[-1].copy()
    calm["volatility"], calm["abs_cost_move"] = 0.001, 0.001
    stormy = panel.iloc[-1].copy()
    stormy["volatility"], stormy["abs_cost_move"] = 0.5, 0.5
    assert calibrator.band(6.5, stormy).width > calibrator.band(6.5, calm).width


def test_calibrator_rejects_mismatched_lengths(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="same length"):
        ConditionalIntervalCalibrator().fit(np.zeros(5), panel)


def test_calibrator_rejects_insufficient_residuals(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not enough usable residuals"):
        ConditionalIntervalCalibrator().fit(np.full(len(panel), np.nan), panel)


def test_calibrator_requires_fit_before_band(panel: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ConditionalIntervalCalibrator().band(6.5, panel.iloc[-1])


@pytest.mark.parametrize("kwargs", [{"nominal": 1.0}, {"nominal": 0.2}, {"min_samples": 5}])
def test_calibrator_rejects_invalid_configuration(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ConditionalIntervalCalibrator(**kwargs)


def test_calibrator_sigma_falls_back_on_missing_features(panel: pd.DataFrame) -> None:
    residuals = np.random.default_rng(5).normal(0.0, 0.02, len(panel))
    calibrator = ConditionalIntervalCalibrator().fit(residuals, panel)
    row = panel.iloc[-1].copy()
    row["volatility"] = np.nan
    assert calibrator.sigma(row) == pytest.approx(calibrator.floor_)
