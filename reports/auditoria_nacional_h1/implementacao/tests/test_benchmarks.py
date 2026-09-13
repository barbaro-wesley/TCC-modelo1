from __future__ import annotations

import numpy as np
import pytest
from benchmarks import gbm
from benchmarks import lstm
from benchmarks.classical import arima_forecast, arimax_forecast, ma_predict, naive_predict


def test_naive_and_moving_average_predictions():
    history = np.array([1.0, 2.0, 3.0, 4.0])
    assert naive_predict(history, n=3).tolist() == [4.0, 4.0, 4.0]
    assert ma_predict(history, window=3, n=2).tolist() == [3.0, 3.0]
    assert ma_predict(history[:2], window=10, n=1)[0] == 1.5


def test_arima_forecast_fit_and_incremental_append_are_finite():
    rng = np.random.default_rng(3)
    history = 4.0 + np.cumsum(rng.normal(0, 0.02, 30))
    first, model = arima_forecast(history, steps=2)
    assert first.shape == (2,)
    assert np.isfinite(first).all()
    extended = np.r_[history, history[-1] + 0.01]
    second, updated = arima_forecast(extended, steps=1, model=model)
    assert second.shape == (1,)
    assert np.isfinite(second[0])
    assert updated is not None


def test_arimax_forecast_fit_and_append_are_finite():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(32, 2))
    y = 5.0 + np.cumsum(0.01 + 0.005 * X[:, 0])
    future = X[-1:]
    first, model = arimax_forecast(y, X, future, steps=1)
    assert first.shape == (1,)
    assert np.isfinite(first[0])
    X2 = np.vstack([X, X[-1]])
    y2 = np.r_[y, y[-1] + 0.01]
    second, _ = arimax_forecast(y2, X2, X2[-1:], steps=1, model=model)
    assert np.isfinite(second[0])


def test_lightgbm_and_xgboost_fit_predict_small_dataset():
    if gbm.lgb is None or gbm.xgb is None:
        pytest.skip("optional GBM benchmark dependencies are not installed")
    rng = np.random.default_rng(9)
    X = rng.normal(size=(40, 3))
    y = 2 * X[:, 0] - X[:, 1]
    lgb_model = gbm.fit_lgbm(X, y)
    assert gbm.predict_lgbm(lgb_model, X[:3]).shape == (3,)
    xgb_model = gbm.fit_xgb(X, y)
    assert np.asarray(xgb_model.predict(X[:3])).shape == (3,)
    assert gbm.fit_lgbm(X, y, model=lgb_model) is lgb_model
    assert gbm.fit_xgb(X, y, model=xgb_model) is xgb_model


def test_gbm_reports_missing_optional_libraries(monkeypatch):
    monkeypatch.setattr(gbm, "lgb", None)
    with pytest.raises(RuntimeError, match="lightgbm"):
        gbm.fit_lgbm(np.ones((2, 1)), np.ones(2))
    monkeypatch.setattr(gbm, "xgb", None)
    with pytest.raises(RuntimeError, match="xgboost"):
        gbm.fit_xgb(np.ones((2, 1)), np.ones(2))


def test_lstm_sequence_shapes_and_empty_case():
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    Xs, ys = lstm.make_sequences(X, y, seq_len=3)
    assert Xs.shape == (7, 3, 2)
    assert ys.shape == (7,)
    assert ys[0] == 3.0
    empty_X, empty_y = lstm.make_sequences(X[:2], y[:2], seq_len=3)
    assert empty_X.shape == (0, 3, 2)
    assert empty_y.shape == (0,)


def test_tiny_lstm_forward_fit_and_predict():
    if lstm.torch is None:
        pytest.skip("optional torch benchmark dependency is not installed")
    model = lstm.TinyLSTM(n_feat=2, hidden=3)
    output = model(lstm.torch.zeros((4, 5, 2)))
    assert tuple(output.shape) == (4,)
    rng = np.random.default_rng(10)
    X = rng.normal(size=(18, 2))
    y = X[:, 0] + 0.2 * X[:, 1]
    assert lstm.fit_lstm(X[:8], y[:8], seq_len=3, epochs=1) is None
    fitted = lstm.fit_lstm(X, y, seq_len=3, epochs=2, hidden=3, seed=1)
    point = lstm.predict_lstm(fitted, X, seq_len=3)
    assert np.isfinite(point)
    assert np.isnan(lstm.predict_lstm(None, X, seq_len=3))
    assert np.isnan(lstm.predict_lstm(fitted, X[:2], seq_len=3))
