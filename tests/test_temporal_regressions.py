"""Causal regression tests; synthetic data only, no downloads or retraining."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import panel
from data.build import add_weekly_lags
from eval.intervals import conformal_p10_p90
from eval.metrics import coverage
from eval.temporal import TEMPORAL_PROTOCOL, training_ends, weekly_history_starts
from eval.walkforward import walk_forward_batch, walk_forward_online
from pipeline import forecast


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def monthly():
    return load_script("02_reproducao.py")


@pytest.fixture
def weekly():
    return load_script("03_semanal.py")


class RecordingModel:
    def __init__(self, *args, **kwargs):
        self.seen = []
        self.snapshots = []
        self.n_rules = 0
        self.beta = 0.18

    def update(self, x, y):
        self.seen.append(float(y))
        self.n_rules = 1

    def predict_one(self, x):
        self.snapshots.append(list(self.seen))
        return float(np.mean(self.seen))


def feature_frame(n=180):
    t = np.arange(n, dtype=float)
    df = pd.DataFrame({
        "data": pd.date_range("2018-01-07", periods=n, freq="7D"),
        "revenda": 4 + 0.01 * t + 0.1 * np.sin(t),
        "distribuicao": 3 + 0.008 * t,
        "brent": 50 + 0.1 * t + np.cos(t),
        "usdbrl": 3 + 0.002 * t,
        "ulsd": np.nan,
    })
    df["brent_brl"] = df["brent"] * df["usdbrl"]
    return df


@pytest.mark.parametrize("h", [1, 2, 4, 6, 12])
def test_training_ends_release_only_mature_labels(h):
    ends = training_ends(40, h)
    for t, end in enumerate(ends):
        assert all(k + h <= t for k in range(end))
        assert end == max(0, t - h + 1)


def test_date_release_survives_filtered_rows_and_mixed_datetime_units():
    origins = pd.date_range("2020-01-05", periods=6, freq="7D").delete([2, 3])
    targets = origins + pd.Timedelta(weeks=2)
    ends = training_ends(4, 2, origins.as_unit("us"), targets.as_unit("ns"))
    assert ends.tolist() == [0, 0, 2, 2]
    assert weekly_history_starts(4, origins.as_unit("us")).tolist() == [0, 0, 2, 2]


@pytest.mark.parametrize("h", [1, 6, 12])
def test_monthly_fit_and_online_updates_do_not_see_future(monthly, monkeypatch, h):
    monkeypatch.setattr(monthly, "VSePLKRLS", RecordingModel)
    X = np.arange(80, dtype=float).reshape(40, 2)
    y = np.arange(40, dtype=float) + h
    _, first, _, model = monthly.run_one(X, y, 20, None, horizon=h)
    assert len(model.snapshots[0]) == 20 - h + 1
    for t, seen in enumerate(model.snapshots, start=20):
        assert max(seen) <= t
    changed = y.copy()
    changed[np.arange(40) + h > 25] += 10000
    _, second, _, _ = monthly.run_one(X, changed, 20, None, horizon=h)
    np.testing.assert_allclose(first[:6], second[:6])


def test_monthly_no_updates_keeps_initial_mature_fit(monthly, monkeypatch):
    monkeypatch.setattr(monthly, "VSePLKRLS", RecordingModel)
    _, _, _, model = monthly.run_one(np.arange(80).reshape(40, 2), np.arange(40),
                                     20, None, seed_update=False, horizon=6)
    assert all(len(s) == 15 for s in model.snapshots)


@pytest.mark.parametrize("h", [1, 2, 4])
def test_intervals_invariant_to_unavailable_residuals(h):
    residuals = np.arange(60, dtype=float)
    changed = residuals.copy()
    changed[25 - h + 1:] += 10000
    a = conformal_p10_p90(residuals, np.zeros(60), min_resid=2, horizon=h)
    b = conformal_p10_p90(changed, np.zeros(60), min_resid=2, horizon=h)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x[:26], y[:26], equal_nan=True)
    expected = np.quantile(residuals[:25 - h + 1], 0.9)
    assert a[1][25] == expected


def test_interval_release_uses_actual_target_dates():
    origins = pd.to_datetime(["2020-01-05", "2020-01-12", "2020-03-01"])
    targets = origins + pd.Timedelta(weeks=4)
    lo, hi = conformal_p10_p90(np.array([1., 3., 999.]), np.zeros(3),
                              min_resid=2, horizon=4, origin_dates=origins, target_dates=targets)
    assert lo[-1] == 1.2
    assert hi[-1] == 2.8
    assert np.isnan(lo[1])


def test_coverage_preserves_dates_after_warmup(weekly):
    y = np.arange(100, dtype=float) + 10
    yhat = y.copy()
    yhat[:10] = np.nan
    metrics, *_ = weekly.eval_model("exact", y, yhat, y, 10, horizon=4)
    assert metrics["coverage_p10_p90"] == 1.0
    assert metrics["pinball10"] == 0.0
    with pytest.raises(ValueError, match="equal length"):
        coverage(y[:10], y, y)


@pytest.mark.parametrize("h", [1, 2, 4])
def test_panel_targets_are_calendar_weeks_and_never_cross_gaps(monkeypatch, h):
    df = add_weekly_lags(feature_frame().drop(index=90))
    monkeypatch.setattr(panel, "load_features", lambda: df.copy())
    result = panel.load_panel(h)
    assert ((result.target_date - result.data) == pd.Timedelta(weeks=h)).all()
    dates = set(df.data)
    for origin in result.data:
        assert all(origin + pd.Timedelta(weeks=k) in dates for k in range(h + 1))
    assert result.attrs["feat_cols"] == panel.FEATURE_COLS


def test_future_completeness_does_not_change_past_features(monkeypatch):
    original = add_weekly_lags(feature_frame())
    monkeypatch.setattr(panel, "load_features", lambda: original.copy())
    a = panel.load_panel(1)
    changed = original.copy()
    changed.loc[100:, "brent_l1"] = np.nan
    monkeypatch.setattr(panel, "load_features", lambda: changed.copy())
    b = panel.load_panel(1)
    assert a.attrs["feat_cols"] == b.attrs["feat_cols"]
    cutoff = original.data.iloc[99]
    pd.testing.assert_frame_equal(a[a.data <= cutoff], b[b.data <= cutoff])


def test_feature_windows_restart_after_calendar_gap():
    raw = feature_frame(60).drop(index=25)
    result = add_weekly_lags(raw)
    after = result[result.data > feature_frame(60).data.iloc[25]].iloc[0]
    assert np.isnan(after.revenda_l1)
    assert np.isnan(after.revenda_ma4)
    assert np.isnan(after.revenda_l12)
    original = add_weekly_lags(raw)
    raw.loc[raw.index >= 40, "revenda"] += 1000
    changed = add_weekly_lags(raw)
    pd.testing.assert_frame_equal(original.iloc[:38], changed.iloc[:38])


def test_latest_features_refuses_to_backdate_missing_latest_week(monkeypatch):
    df = add_weekly_lags(feature_frame())
    df.loc[df.index[-1], "revenda_l1"] = np.nan
    monkeypatch.setattr(panel, "load_features", lambda: df)
    with pytest.raises(ValueError, match="origem antiga"):
        panel.latest_features(panel.FEATURE_COLS)


@pytest.mark.parametrize("h", [1, 4])
def test_arimax_production_includes_latest_price_and_requested_horizon(monkeypatch, h):
    df = add_weekly_lags(feature_frame())
    monkeypatch.setattr(forecast, "load_features", lambda: df)
    def predict(prices, exog, future, steps):
        assert prices[-1] == df.revenda.iloc[-1]
        np.testing.assert_allclose(exog[-1], df[panel.ARIMAX_COLS].iloc[-1].to_numpy(float))
        assert steps == h
        assert future.shape == (h, 2)
        return np.arange(h), None
    monkeypatch.setattr(forecast, "arimax_forecast", predict)
    assert forecast._prever_arimax(h) == h - 1


def test_publication_rejects_horizon_with_wrong_ranking():
    with pytest.raises(ValueError, match="ranking proprio"):
        forecast.montar_previsao(4)


@pytest.mark.parametrize("h", [1, 2, 4])
def test_weekly_online_helpers_release_only_mature_labels(weekly, monkeypatch, h):
    X = np.arange(120, dtype=float).reshape(60, 2)
    y = np.arange(60) + h
    monkeypatch.setattr(weekly, "VSePLKRLS", RecordingModel)
    _, model, _ = weekly.run_vsepl(X, y, 20, h)
    for t, seen in enumerate(model.snapshots, start=20):
        assert max(seen) <= t
    result = walk_forward_online(RecordingModel, X, y, 20, horizon=h)
    for t, seen in enumerate(result["model"].snapshots, start=20):
        assert max(seen) <= t
    assert max(result["model"].seen) <= len(y) - 1


@pytest.mark.parametrize("h", [1, 4])
def test_batch_helpers_never_receive_future_labels(weekly, monkeypatch, h):
    X = np.arange(120, dtype=float).reshape(60, 2)
    y = np.arange(60) + h
    class BatchModel:
        def __init__(self, seen):
            self.seen = seen
        def predict(self, query):
            t = int(query[0, 0] / 2)
            assert max(self.seen) <= t
            return [np.mean(self.seen)]
    monkeypatch.setattr(weekly, "fit_lgbm", lambda x, y: BatchModel(y))
    a, _ = weekly.run_gbm("lgbm", X, y, 25, h, refit_every=1)
    changed = y.copy()
    changed[y > 30] += 10000
    # Later predictions can change; inspect causality at earlier origins.
    monkeypatch.setattr(weekly, "fit_lgbm", lambda x, y: types.SimpleNamespace(predict=lambda q: [np.mean(y)]))
    b, _ = weekly.run_gbm("lgbm", X, changed, 25, h, refit_every=1)
    np.testing.assert_allclose(a[:31], b[:31], equal_nan=True)
    def fit_predict(x, labels, query, **kwargs):
        assert labels.max() <= query[0, 0] / 2
        return float(labels.mean()), object()
    walk_forward_batch(fit_predict, X, y, 25, horizon=h)


@pytest.mark.parametrize("h", [1, 2, 4])
def test_lstm_inference_window_is_independent_of_label_delay(weekly, monkeypatch, h):
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = np.arange(60) + h
    def fit(x, labels, **kwargs):
        return labels.copy()
    def predict(labels, x, seq_len):
        t = len(x)
        assert labels.max() <= t
        assert x[-seq_len:, 0].tolist() == list(range(t - seq_len, t))
        return float(x[-1, 0])
    monkeypatch.setattr(weekly, "fit_lstm", fit)
    monkeypatch.setattr(weekly, "predict_lstm", predict)
    result = weekly.run_lstm_wf(X, y, 25, h, seq_len=8)
    np.testing.assert_allclose(result[25:], np.arange(24, 59))


def test_lstm_subprocess_entrypoint_uses_same_window(monkeypatch, tmp_path):
    fake = types.ModuleType("benchmarks.lstm")
    fake.fit_lstm = lambda x, y, **kw: y.copy()
    def predict(labels, x, seq_len):
        assert labels.max() <= len(x)
        return float(x[-1, 0])
    fake.predict_lstm = predict
    monkeypatch.setitem(sys.modules, "benchmarks.lstm", fake)
    script = load_script("_lstm_wf.py")
    path = tmp_path / "input.npz"
    dates = pd.date_range("2020-01-05", periods=60, freq="7D")
    np.savez(path, X=np.arange(60).reshape(-1, 1), y=np.arange(60) + 4,
             origin_dates=dates.to_numpy(), target_dates=(dates + pd.Timedelta(weeks=4)).to_numpy())
    monkeypatch.setattr(sys, "argv", ["_lstm_wf.py", str(path), "25", "4", "26", "8"])
    script.main()
    np.testing.assert_allclose(np.load(tmp_path / "input_yhat.npy")[25:], np.arange(24, 59))


def test_arima_restarts_instead_of_compressing_missing_weeks(weekly, monkeypatch):
    dates = pd.date_range("2020-01-05", periods=40, freq="7D").delete(20)
    prices = np.arange(len(dates), dtype=float)
    calls = []
    def predict(history, steps, model=None):
        calls.append(history.copy())
        return np.repeat(history[-1], steps), object()
    monkeypatch.setattr(weekly, "arima_forecast", predict)
    weekly.run_arima(prices, 10, 1, dates=dates)
    assert any(history[0] == 20 for history in calls)
    assert all(not (history[0] < 20 and history[-1] >= 20) for history in calls)


def test_lstm_training_sequences_do_not_bridge_gaps():
    from benchmarks.sequences import make_sequences
    dates = pd.date_range("2020-01-05", periods=20, freq="7D").delete(10)
    X = np.arange(len(dates)).reshape(-1, 1)
    sequences, labels = make_sequences(X, np.arange(len(dates)), 4, dates=dates)
    assert not set(range(10, 14)).intersection(labels)
    for seq, target in zip(sequences, labels):
        np.testing.assert_array_equal(seq[:, 0], np.arange(int(target) - 4, int(target)))


def test_old_metrics_cannot_be_used_for_publication(monkeypatch, tmp_path):
    monkeypatch.setattr(forecast, "RES", tmp_path)
    old = pd.DataFrame({"model": ["ARIMA"], "rmse": [0.07], "horizon": [1]})
    old.to_csv(tmp_path / "semanal_benchmarks.csv", index=False)
    with pytest.raises(ValueError, match="protocolo antigo"):
        forecast.ranking_h1()
    old["temporal_protocol"] = TEMPORAL_PROTOCOL
    old.to_csv(tmp_path / "semanal_benchmarks.csv", index=False)
    assert forecast.ranking_h1().model.iloc[0] == "ARIMA"


def test_old_residuals_cannot_calibrate_new_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(forecast, "RES", tmp_path)
    old = pd.DataFrame({"y": [1.0], "ARIMA": [0.5]})
    old.to_csv(tmp_path / "walkforward_preds_h1.csv", index=False)
    with pytest.raises(ValueError, match="protocolo antigo"):
        forecast.residuos_walkforward("ARIMA")
    old["temporal_protocol"] = TEMPORAL_PROTOCOL
    old.to_csv(tmp_path / "walkforward_preds_h1.csv", index=False)
    np.testing.assert_array_equal(forecast.residuos_walkforward("ARIMA"), [0.5])


def test_weekly_main_smoke_exports_aligned_artifacts(weekly, monkeypatch, tmp_path):
    import json

    raw = feature_frame(200).drop(index=95)
    df = add_weekly_lags(raw)
    monkeypatch.setattr(panel, "load_features", lambda: df.copy())
    monkeypatch.setattr(forecast, "load_features", lambda: df.copy())
    monkeypatch.setattr(weekly, "SKIP_LSTM", True)
    monkeypatch.setattr(weekly, "RES", tmp_path / "results")
    monkeypatch.setattr(weekly, "FIG", tmp_path / "figures")
    monkeypatch.setattr(weekly, "REP", tmp_path)
    monkeypatch.setattr(forecast, "RES", tmp_path / "results")
    monkeypatch.setattr(weekly, "VSePLKRLS", RecordingModel)
    monkeypatch.setattr(forecast, "VSePLKRLS", RecordingModel)

    def fit(x, labels):
        return types.SimpleNamespace(predict=lambda q: np.repeat(np.mean(labels), len(q)))
    monkeypatch.setattr(weekly, "fit_lgbm", fit)
    monkeypatch.setattr(weekly, "fit_xgb", fit)
    monkeypatch.setattr(forecast, "fit_lgbm", fit)
    monkeypatch.setattr(forecast, "fit_xgb", fit)
    def arima(prices, steps, model=None):
        return np.repeat(prices[-1], steps), object()
    def arimax(prices, exog, future, steps, model=None):
        return arima(prices, steps, model)
    monkeypatch.setattr(weekly, "arima_forecast", arima)
    monkeypatch.setattr(weekly, "arimax_forecast", arimax)
    monkeypatch.setattr(forecast, "arima_forecast", arima)
    monkeypatch.setattr(forecast, "arimax_forecast", arimax)
    weekly.main()
    for h in (1, 2, 4):
        wf = pd.read_csv(tmp_path / "results" / f"walkforward_preds_h{h}.csv",
                         parse_dates=["data", "target_date"])
        assert (wf.target_date - wf.data).eq(pd.Timedelta(weeks=h)).all()
        assert wf.temporal_protocol.eq(TEMPORAL_PROTOCOL).all()
        importance = json.loads((tmp_path / "results" / f"importancia_lgbm_h{h}.json").read_text())
        assert set(importance) == set(panel.FEATURE_COLS)
    production = forecast.montar_previsao()
    assert production["ultima_semana_observada"] == str(df.data.iloc[-1].date())
    assert production["semana_prevista"] == str((df.data.iloc[-1] + pd.Timedelta(weeks=1)).date())
    assert production["temporal_protocol"] == TEMPORAL_PROTOCOL


def test_real_feature_cache_is_rebuilt_causally(monkeypatch, tmp_path):
    raw = feature_frame(80).drop(index=40)
    # A legacy cache can contain stale positional lags; load_features must not
    # trust those columns when the raw observation dates have a missing week.
    raw["revenda_l1"] = 12345.0
    raw.to_csv(tmp_path / "semanal_s10_features.csv", index=False)
    monkeypatch.setattr(panel, "PROC", tmp_path)
    rebuilt = panel.load_features()
    after_gap = raw.data.iloc[40]
    assert np.isnan(rebuilt.loc[rebuilt.data == after_gap, "revenda_l1"].iloc[0])
    assert rebuilt.revenda_l1.dropna().max() < 10


def test_initial_predictions_are_not_calibration_residuals(weekly):
    y = np.arange(60, dtype=float) + 10
    metrics, residuals, lo, hi, *_ = weekly.eval_model("exact", y, y.copy(), y, 20)
    assert np.isnan(residuals[:20]).all()
    assert np.isnan(lo[:40]).all()
    assert metrics["coverage_p10_p90"] == 1.0


def test_monthly_release_uses_original_dates_after_missing_pairs(monthly, monkeypatch):
    monkeypatch.setattr(monthly, "VSePLKRLS", RecordingModel)
    original = pd.date_range("2000-01-01", periods=70, freq="MS")
    indices = np.delete(np.arange(58), [3, 8, 9, 31])
    origins = original[indices]
    targets = original[indices + 12]
    y = (indices + 12).astype(float)
    _, _, _, model = monthly.run_one(np.column_stack([indices, indices]), y, 25, None,
                                     horizon=12, origin_dates=origins, target_dates=targets)
    for pos, seen in enumerate(model.snapshots, start=25):
        assert max(seen) <= indices[pos]
