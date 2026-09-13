from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vsepl_krls.model import VSePLKRLSConfig


def test_download_script_main_reports_gate(monkeypatch, capsys, load_script, tmp_path):
    script = load_script("01_download.py")
    raw_file = tmp_path / "raw.xlsx"
    raw_file.write_bytes(b"x")
    monkeypatch.setattr(script, "download_all", lambda: {"mensal": raw_file})
    monkeypatch.setattr(
        script,
        "save_processed",
        lambda: {
            "gate": {
                "ok": True,
                "observed": {"n": 90, "mean": 3.1},
                "expected": {"n": 90, "mean": 3.1},
                "diffs": {"n": 0, "mean": 0.0},
            },
            "monthly": pd.DataFrame(index=range(2)),
            "weekly": pd.DataFrame(index=range(3)),
            "paper": pd.DataFrame(index=range(1)),
        },
    )
    script.main()
    output = capsys.readouterr().out
    assert "Arquivos baixados" in output
    assert "ok = True" in output
    assert "Semanal completo: 3" in output


def test_reproduction_run_one_returns_finite_contract(load_script):
    script = load_script("02_reproducao.py")
    X = np.column_stack([np.linspace(0, 1, 25), np.linspace(1, 2, 25)])
    y = 2.0 + 0.2 * X[:, 0]
    metrics, preds, yte, model = script.run_one(
        X,
        y,
        n_train=18,
        cfg=VSePLKRLSConfig(use_variable_step=False, d_max=5),
    )
    assert len(preds) == len(yte) == 7
    assert np.isfinite(preds).all()
    assert metrics["n_rules"] == model.n_rules
    assert {"rmse", "mae", "ndei_full", "beta_final"}.issubset(metrics)


def test_reproduction_experiment_builds_expected_configuration(monkeypatch, load_script):
    script = load_script("02_reproducao.py")
    frame = pd.DataFrame({"revenda": np.arange(90.0), "distribuicao": np.arange(90.0)})
    monkeypatch.setattr(script.pd, "read_csv", lambda *args, **kwargs: frame)
    monkeypatch.setattr(
        script,
        "make_supervised",
        lambda *args, **kwargs: (np.ones((89, 2)), np.ones(89), np.arange(89)),
    )

    class Model:
        history_n_rules = [1]
        history_beta = [0.2]

    monkeypatch.setattr(
        script,
        "run_one",
        lambda X, y, n_train, cfg: (
            {"rmse": 1.0, "mae": 1.0, "ndei": 1.0, "n_rules": 1, "beta_final": 0.2},
            np.array([1.0]),
            np.array([1.0]),
            Model(),
        ),
    )
    result = script.experiment(1, "texto", True)
    assert result["horizon"] == 1
    assert result["convention"] == "texto"
    assert result["variable_step"] is True
    assert result["n_train"] == 72
    assert result["n_rules_path"] == [1]


@pytest.mark.integration
def test_reproduction_main_writes_table_json_plot_and_report(monkeypatch, load_script, tmp_path):
    script = load_script("02_reproducao.py")
    proc = tmp_path / "processed"
    res = tmp_path / "results"
    fig = tmp_path / "figures"
    rep = tmp_path / "reports"
    proc.mkdir()
    pd.DataFrame(
        {
            "data": pd.date_range("2012-12-01", periods=3, freq="MS"),
            "revenda": [3.0, 3.1, 3.2],
            "distribuicao": [2.8, 2.9, 3.0],
        }
    ).to_csv(proc / "mensal_s10_artigo.csv", index=False)

    def fake_experiment(horizon, convention, vs):
        base = 0.05 + 0.001 * horizon + (0.001 if convention == "texto" else 0.0)
        return {
            "horizon": horizon,
            "convention": convention,
            "variable_step": vs,
            "n_pairs": 80,
            "n_train": 60,
            "n_test": 2,
            "metrics": {
                "rmse": base,
                "mae": base,
                "ndei": base,
                "ndei_full": base,
                "n_rules": 1.0,
                "beta_final": 0.2,
            },
            "preds": [3.1, 3.2],
            "y_test": [3.1, 3.2],
            "n_rules_path": [1],
            "beta_path": [0.2],
        }

    monkeypatch.setattr(script, "PROC", proc)
    monkeypatch.setattr(script, "RES", res)
    monkeypatch.setattr(script, "FIG", fig)
    monkeypatch.setattr(script, "REP", rep)
    monkeypatch.setattr(script, "experiment", fake_experiment)
    script.main()
    table = pd.read_csv(res / "reproducao_mensal.csv")
    assert len(table) == 12
    assert (res / "reproducao_mensal.json").exists()
    assert (fig / "reproducao_h1.png").exists()
    assert "Bloco 1" in (rep / "01_reproducao.md").read_text(encoding="utf-8")


def test_weekly_scaling_panel_and_simple_predictors(monkeypatch, load_script, tmp_path):
    script = load_script("03_semanal.py")
    Xtr = np.array([[0.0, 2.0], [2.0, 2.0]])
    Xte = np.array([[1.0, 5.0]])
    scaled_tr, scaled_te, lo, span = script.minmax_train(Xtr, Xte)
    assert np.allclose(scaled_tr[:, 0], [0.0, 1.0])
    assert scaled_te[0, 0] == 0.5
    assert span[1] == 1.0
    frozen, _, _ = script.scale_frozen(np.vstack([Xtr, Xte]), n_min=2)
    assert frozen[-1, 1] == 1.25

    dates = pd.date_range("2021-01-03", periods=30, freq="7D")
    frame = pd.DataFrame(
        {
            "data": dates,
            "revenda": np.linspace(4.0, 5.0, 30),
            "revenda_l1": np.linspace(3.9, 4.9, 30),
            "brent_l1": np.linspace(50, 60, 30),
            "usdbrl_l1": np.linspace(4, 5, 30),
        }
    )
    frame.to_csv(tmp_path / "semanal_s10_features.csv", index=False)
    monkeypatch.setattr(script, "PROC", tmp_path)
    panel = script.load_panel(horizon=2)
    assert len(panel) == 28
    assert panel.loc[0, "y"] == pytest.approx(frame.loc[2, "revenda"])
    assert script.run_naive(np.array([1.0, 2.0, 3.0]), 1).tolist()[1:] == [2.0, 3.0]
    assert np.isfinite(script.run_ma(np.arange(1.0, 8.0), 2, window=2)[2:]).all()


def test_eval_model_aligns_conformal_coverage(load_script):
    script = load_script("03_semanal.py")
    y = np.linspace(4.0, 6.0, 80)
    residual_pattern = np.tile(np.array([-0.1, -0.05, 0.0, 0.05, 0.1]), 16)
    yhat = y - residual_pattern
    y_prev = y - 0.02
    metrics, residuals, lo, hi, mask, errors = script.eval_model(
        "modelo",
        y,
        yhat,
        y_prev,
        n_min=10,
    )
    assert metrics["coverage_p10_p90"] > 0.5
    assert 0.0 <= metrics["coverage_p10_p90"] <= 1.0
    assert len(residuals) == len(lo) == len(hi) == len(mask)
    assert len(errors) == int(mask.sum())


def test_weekly_model_runners_have_leak_safe_shapes(monkeypatch, load_script):
    script = load_script("03_semanal.py")
    X = np.column_stack([np.linspace(0, 1, 28), np.linspace(1, 2, 28)])
    y = np.linspace(4.0, 5.0, 28)
    price = y - 0.01
    vsepl, model, scaler = script.run_vsepl(X, y, n_min=5, horizon=2)
    assert np.isnan(vsepl[:5]).all()
    assert np.isfinite(vsepl[5:]).all()
    assert model.n_seen == len(y) - 2
    assert len(scaler) == 2

    arima_calls = []

    def fake_arima(history, steps, model=None):
        arima_calls.append((len(history), model is None))
        return np.repeat(history[-1], steps), object()

    monkeypatch.setattr(script, "arima_forecast", fake_arima)
    arima = script.run_arima(price, n_min=5, horizon=2, refit_every=3)
    assert np.isnan(arima[:5]).all()
    assert np.isfinite(arima[5:]).all()
    assert any(is_new for _, is_new in arima_calls)

    arimax_calls = []

    def fake_arimax(history, Xhist, Xfuture, steps, model=None):
        arimax_calls.append((len(history), model is None))
        return np.repeat(history[-1], steps), object()

    monkeypatch.setattr(script, "arimax_forecast", fake_arimax)
    arimax = script.run_arimax(price, X, n_min=5, horizon=2, refit_every=3)
    assert np.isfinite(arimax[5:]).all()
    assert arimax_calls

    monkeypatch.setattr(script, "fit_lgbm", lambda Xtr, ytr: FakePredictor(float(np.mean(ytr))))
    gbm_pred, fitted = script.run_gbm("lgbm", X, y, n_min=5, horizon=2, refit_every=3)
    assert fitted is not None
    assert np.isfinite(gbm_pred).sum() > 0
    monkeypatch.setattr(script, "fit_xgb", lambda Xtr, ytr: FakePredictor(float(np.mean(ytr))))
    xgb_pred, _ = script.run_gbm("xgb", X, y, n_min=5, horizon=2, refit_every=3)
    assert np.isfinite(xgb_pred).sum() > 0

    monkeypatch.setattr(script, "fit_lstm", lambda *args, **kwargs: object())
    monkeypatch.setattr(script, "predict_lstm", lambda *args, **kwargs: 4.5)
    lstm = script.run_lstm_wf(X, y, n_min=5, horizon=2, refit_every=3, seq_len=2)
    assert np.isfinite(lstm).sum() > 0


def test_markdown_table_and_forecast_plot(load_script, tmp_path):
    script = load_script("03_semanal.py")
    table = script.md_table(pd.DataFrame([{"modelo": "A", "rmse": 0.123456}]))
    assert "| modelo | rmse |" in table
    assert "0.12346" in table
    dates = pd.date_range("2024-01-01", periods=3)
    path = tmp_path / "forecast.png"
    script.plot_forecast(
        dates,
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.1]),
        np.array([0.9, 1.7, 2.8]),
        np.array([1.3, 2.1, 3.3]),
        "teste",
        path,
    )
    assert path.exists()


class FakePredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.repeat(self.value, len(X))


class FakeVS:
    n_rules = 2

    def __init__(self):
        self.updated = []

    def update(self, x, y):
        self.updated.append((np.asarray(x), y))

    def predict_one(self, x):
        return 8.5


def test_forecast_next_model_dispatches_every_supported_model(monkeypatch, load_script):
    script = load_script("03_semanal.py")
    price = np.array([4.0, 4.1, 4.2, 4.3])
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.linspace(4.0, 5.0, 12)
    x_next = np.array([25.0, 26.0])
    Xex = np.arange(8, dtype=float).reshape(4, 2)

    monkeypatch.setattr(script, "arima_forecast", lambda *args, **kwargs: (np.array([7.1]), object()))
    monkeypatch.setattr(script, "arimax_forecast", lambda *args, **kwargs: (np.array([7.2]), object()))
    monkeypatch.setattr(script, "fit_lgbm", lambda *args, **kwargs: FakePredictor(7.3))
    monkeypatch.setattr(script, "fit_xgb", lambda *args, **kwargs: FakePredictor(7.4))
    monkeypatch.setattr(script, "fit_lstm", lambda *args, **kwargs: object())
    monkeypatch.setattr(script, "predict_lstm", lambda *args, **kwargs: 7.5)

    assert script.forecast_next_model("ARIMA", price, X, y, x_next, Xex)[0] == 7.1
    assert script.forecast_next_model("ARIMAX", price, X, y, x_next, Xex)[0] == 7.2
    assert script.forecast_next_model("naive", price, X, y, x_next, Xex)[0] == 4.3
    assert script.forecast_next_model("media_movel", price, X, y, x_next, Xex)[0] == pytest.approx(4.15)
    assert script.forecast_next_model("LightGBM", price, X, y, x_next, Xex)[0] == 7.3
    assert script.forecast_next_model("XGBoost", price, X, y, x_next, Xex)[0] == 7.4
    assert script.forecast_next_model("LSTM", price, X, y, x_next, Xex)[0] == 7.5

    vs = FakeVS()
    point, meta = script.forecast_next_model(
        "VS-ePL-KRLS",
        price,
        X,
        y,
        x_next,
        Xex,
        vs_model=vs,
        vs_scaler=(np.zeros(2), np.ones(2)),
    )
    assert point == 8.5
    assert meta == {"n_regras": 2}
    assert len(vs.updated) == 1
    with pytest.raises(ValueError, match="nao suportado"):
        script.forecast_next_model("desconhecido", price, X, y, x_next, Xex)


@pytest.mark.integration
def test_production_script_rejects_mismatch_and_writes_valid_report(monkeypatch, load_script, tmp_path):
    script = load_script("04_producao.py")
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    results.mkdir()
    reports.mkdir()
    pd.DataFrame(
        [
            {
                "horizon": 1,
                "model": "ARIMA",
                "rmse": 0.1,
                "mae": 0.05,
                "smape": 1.0,
                "dir_acc": 0.6,
                "coverage_p10_p90": 0.8,
            },
            {
                "horizon": 1,
                "model": "naive",
                "rmse": 0.2,
                "mae": 0.1,
                "smape": 2.0,
                "dir_acc": 0.0,
                "coverage_p10_p90": 0.8,
            },
        ]
    ).to_csv(results / "semanal_benchmarks.csv", index=False)
    payload = {
        "modelo": "VS-ePL-KRLS",
        "ultima_semana_observada": "2026-08-09",
        "preco_observado_ultima_semana": 6.9,
        "previsao_pontual": 6.91,
        "p10": 6.85,
        "p90": 6.97,
        "probabilidades": {"p_alta": 0.2, "p_estavel": 0.6, "p_queda": 0.2},
    }
    (results / "previsao_proxima_semana.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(script, "RES", results)
    monkeypatch.setattr(script, "REP", reports)
    with pytest.raises(RuntimeError, match="inconsistente"):
        script.main()

    payload["modelo"] = "ARIMA"
    (results / "previsao_proxima_semana.json").write_text(json.dumps(payload), encoding="utf-8")
    script.main()
    report = (reports / "03_producao.md").read_text(encoding="utf-8")
    assert "Recomendado (h=1 semana, criterio RMSE walk-forward): ARIMA" in report
    assert "Previsao pontual: 6.91" in report


def test_lstm_subprocess_script_writes_predictions(monkeypatch, load_script, tmp_path):
    script = load_script("_lstm_wf.py")
    path = tmp_path / "sample.npz"
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.linspace(1.0, 2.0, 20)
    np.savez(path, X=X, y=y)
    monkeypatch.setattr(script, "fit_lstm", lambda *args, **kwargs: object())
    monkeypatch.setattr(script, "predict_lstm", lambda *args, **kwargs: 1.5)
    monkeypatch.setattr(sys, "argv", ["_lstm_wf.py", str(path), "5", "1", "4", "2"])
    script.main()
    output = np.load(tmp_path / "sample_yhat.npy")
    assert output.shape == y.shape
    assert np.isfinite(output).sum() == 10


def test_production_payload_validation_rejects_invalid_contract(load_script):
    script = load_script("04_producao.py")
    valid = {
        "modelo": "ARIMA",
        "ultima_semana_observada": "2026-08-09",
        "preco_observado_ultima_semana": 6.9,
        "previsao_pontual": 6.91,
        "p10": 6.85,
        "p90": 6.97,
        "probabilidades": {"p_alta": 0.2, "p_estavel": 0.6, "p_queda": 0.2},
    }
    script.validate_forecast_payload(valid, "ARIMA")
    with pytest.raises(RuntimeError, match="incompleto"):
        script.validate_forecast_payload({"modelo": "ARIMA"}, "ARIMA")
    with pytest.raises(RuntimeError, match="inconsistente"):
        script.validate_forecast_payload(valid, "ARIMAX")
    with pytest.raises(RuntimeError, match="dentro do intervalo"):
        script.validate_forecast_payload({**valid, "p10": 7.0}, "ARIMA")
    with pytest.raises(RuntimeError, match="somam 1"):
        script.validate_forecast_payload(
            {**valid, "probabilidades": {"p_alta": 0.2, "p_estavel": 0.2, "p_queda": 0.2}},
            "ARIMA",
        )


@pytest.mark.integration
def test_weekly_main_runs_end_to_end_with_fast_models(monkeypatch, load_script, tmp_path):
    script = load_script("03_semanal.py")
    proc = tmp_path / "processed"
    res = tmp_path / "results"
    fig = tmp_path / "figures"
    rep = tmp_path / "reports"
    proc.mkdir()
    feat_cols = ["revenda_l1", "brent_l1", "usdbrl_l1"]
    full_dates = pd.date_range("2024-01-07", periods=91, freq="7D")
    full = pd.DataFrame(
        {
            "data": full_dates,
            "revenda": np.linspace(5.0, 6.0, 91),
            "revenda_l1": np.linspace(4.99, 5.99, 91),
            "brent_l1": np.linspace(70.0, 80.0, 91),
            "usdbrl_l1": np.linspace(4.8, 5.2, 91),
        }
    )
    full.to_csv(proc / "semanal_s10_features.csv", index=False)

    def fake_panel(horizon):
        n = 90
        y = np.linspace(5.1, 6.0, n)
        frame = pd.DataFrame(
            {
                "data": full_dates[:n],
                "revenda": y - 0.01,
                "y_prev": y - 0.01,
                "y": y,
                "revenda_l1": np.linspace(5.0, 5.9, n),
                "brent_l1": np.linspace(70.0, 79.0, n),
                "usdbrl_l1": np.linspace(4.8, 5.1, n),
            }
        )
        frame.attrs["feat_cols"] = feat_cols
        return frame

    class FastVS:
        n_rules = 1
        beta = 0.2

        def predict_one(self, x):
            return 5.5

        def update(self, x, y):
            return y

    def fake_vsepl(X, y, n_min, horizon):
        return y + 0.4, FastVS(), (np.zeros(X.shape[1]), np.ones(X.shape[1]))

    monkeypatch.setattr(script, "PROC", proc)
    monkeypatch.setattr(script, "RES", res)
    monkeypatch.setattr(script, "FIG", fig)
    monkeypatch.setattr(script, "REP", rep)
    monkeypatch.setattr(script, "load_panel", fake_panel)
    monkeypatch.setattr(script, "run_vsepl", fake_vsepl)
    monkeypatch.setattr(script, "run_naive", lambda price, n_min: fake_panel(1)["y"].to_numpy() - 0.01)
    monkeypatch.setattr(script, "run_ma", lambda price, n_min, window: fake_panel(1)["y"].to_numpy() + 0.1)
    monkeypatch.setattr(script, "run_arima", lambda price, n_min, horizon, refit: fake_panel(horizon)["y"].to_numpy())
    monkeypatch.setattr(script, "run_arimax", lambda price, Xex, n_min, horizon, refit: fake_panel(horizon)["y"].to_numpy() + 0.02)

    def fake_gbm(kind, X, y, n_min, horizon, refit_every):
        offset = 0.2 if kind == "lgbm" else 0.25
        return y + offset, FakePredictor(float(np.mean(y)))

    monkeypatch.setattr(script, "run_gbm", fake_gbm)

    def fake_subprocess_run(args, cwd):
        npz_path = script.Path(args[2])
        values = np.load(npz_path)["y"] + 0.3
        out = npz_path.with_name(npz_path.stem + "_yhat.npy")
        np.save(out, values)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(
        script,
        "forecast_next_model",
        lambda name, *args, **kwargs: (6.01, {"origem_teste": True}),
    )
    script.main()
    payload = json.loads((res / "previsao_proxima_semana.json").read_text(encoding="utf-8"))
    assert payload["modelo"] == "ARIMA"
    assert payload["previsao_pontual"] == 6.01
    assert payload["origem_teste"] is True
    assert (res / "semanal_benchmarks.csv").exists()
    assert (rep / "02_semanal.md").exists()
    assert len(list(fig.glob("*.png"))) >= 6


def test_s10_selection_helpers_are_deterministic_and_validate_shapes(load_script):
    script = load_script("05_s10_model_selection.py")
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predictions = np.column_stack(
        [actual, actual + 0.5, np.repeat(2.5, 4), actual - 0.2]
    )
    weights = script._convex_weights(actual, predictions, resolution=5)
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] == 1.0
    with pytest.raises(ValueError, match="shape"):
        script._convex_weights(actual, predictions[:-1])
    with pytest.raises(ValueError, match="finite"):
        script._convex_weights(actual, predictions * np.nan)

    row = script._metric_row("perfect", actual, actual, actual + 1.0, horizon=1)
    assert row["rmse"] == 0.0
    assert row["n"] == 4


def test_s10_production_approval_falls_back_to_arima(load_script):
    script = load_script("06_train_s10_production.py")
    base = {
        "selected_production_model": "ensemble",
        "comparison": [
            {"model": "ARIMA", "rmse": 1.0},
            {"model": "ensemble", "rmse": 1.1},
        ],
    }
    assert script._approved_primary(base) == "ARIMA"
    base["comparison"][1]["rmse"] = 1.01
    assert script._approved_primary(base) == "ensemble"
    base["selected_production_model"] = "missing"
    with pytest.raises(RuntimeError, match="absent"):
        script._approved_primary(base)
    table = script._markdown_table(pd.DataFrame([{"model": "A", "rmse": 0.123456}]))
    assert "| model | rmse |" in table
    assert "0.12346" in table


def test_next_challenger_helpers_are_causal_and_bounded(monkeypatch, load_script):
    script = load_script("08_s10_next_challenger.py")
    n = 70
    prices = np.linspace(5.0, 6.0, n)
    data = script.S10Supervised(
        x=prices[:, None],
        target_price=prices + 0.01,
        origin_price=prices,
        dates=pd.date_range("2024-01-01", periods=n, freq="7D").to_numpy(),
        target_dates=pd.date_range("2024-01-08", periods=n, freq="7D").to_numpy(),
        feature_names=("price_t",),
        horizon=1,
    )
    histories = []

    def fake_arima(history, steps, model=None):
        histories.append(np.asarray(history).copy())
        return np.array([history[-1]]), object()

    monkeypatch.setattr(script, "arima_forecast", fake_arima)
    predictions = script._causal_arima_predictions(
        data, end=60, min_history=52, refit_every=13
    )
    assert np.isnan(predictions[:52]).all()
    assert np.allclose(predictions[52:60], prices[52:60])
    assert [len(history) for history in histories] == list(range(53, 61))
    assert all(history[-1] == pytest.approx(prices[len(history) - 1]) for history in histories)

    frozen = script.S10Candidate(
        candidate_id="frozen",
        feature_set="lags",
        target_mode="delta",
        alpha=0.26,
        beta_initial=0.18,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=0.5,
        kernel_sigma=0.15,
        regularization=1e-3,
    )
    direct = script._targeted_direct_candidates(frozen)
    hybrid = script._targeted_hybrid_candidates(frozen)
    assert len(direct) == 13
    assert len(hybrid) == 9
    assert len({candidate.candidate_id for candidate in direct + hybrid}) == 22
    assert all(candidate.target_mode == "delta" for candidate in hybrid)
    assert max(candidate.max_dictionary_size for candidate in direct) == 40
    conservative = next(
        candidate for candidate in hybrid if candidate.candidate_id == "hybrid_lags_conservative"
    )
    assert conservative.residual_correction_weight == 0.5
    assert conservative.residual_correction_limit == 0.1

    table = script._markdown_table(pd.DataFrame([{"model": "A", "rmse": 0.1234567}]))
    assert "| model | rmse |" in table
    assert "0.123457" in table


def test_s10_runtime_cli_is_read_only_by_default_and_updates_atomically(
    monkeypatch, load_script, tmp_path
):
    script = load_script("07_s10_predict.py")

    class Payload:
        def __init__(self, value):
            self.value = value

        def as_dict(self):
            return self.value

    class FakeBundle:
        def __init__(self):
            self.updates = []
            self.saved = []

        def update_one(self, date, price, *, allow_anomalous_change=False):
            self.updates.append((date, price, allow_anomalous_change))
            return self

        def save(self, path):
            destination = script.Path(path)
            destination.write_bytes(b"model")
            self.saved.append(destination)
            return destination

        def predict_next(self):
            return Payload({"point": 6.9})

        def health(self):
            return Payload({"status": "healthy"})

        def metadata(self):
            return {"artifact_version": "1.0.0"}

    fake = FakeBundle()
    monkeypatch.setattr(
        script.S10ProductionForecaster,
        "load",
        lambda _, expected_sha256=None: fake,
    )
    artifact = tmp_path / "in.joblib"
    artifact.write_bytes(b"input")
    output = tmp_path / "forecast.json"
    base = SimpleNamespace(
        artifact=artifact,
        expected_sha256=None,
        allow_anomalous_change=False,
        update_date=None,
        update_price=None,
        output_artifact=None,
        output=output,
    )
    result = script.run(base)
    assert result["updated"] is False
    assert not fake.updates
    assert json.loads(output.read_text(encoding="utf-8"))["forecast"]["point"] == 6.9

    destination = tmp_path / "next.joblib"
    updated = script.run(
        SimpleNamespace(
            **{
                **vars(base),
                "update_date": "2026-08-16",
                "update_price": 6.91,
                "output_artifact": destination,
            }
        )
    )
    assert updated["updated"] is True
    assert fake.updates == [("2026-08-16", 6.91, False)]
    assert destination.exists()
    with pytest.raises(ValueError, match="together"):
        script.run(SimpleNamespace(**{**vars(base), "update_date": "2026-08-16"}))


def test_s10_shadow_cli_accepts_only_a_frozen_eligible_manifest(load_script, tmp_path):
    script = load_script("09_s10_shadow.py")
    candidate = script.S10Candidate(
        candidate_id="shadow",
        feature_set="dynamics",
        target_mode="delta",
        alpha=0.26,
        beta_initial=0.18,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=0.5,
        kernel_sigma=0.15,
        regularization=1e-3,
        residual_correction_weight=0.5,
        residual_correction_limit=0.1,
    )
    manifest = {
        "holdout_evaluated": False,
        "production_promotion_allowed": False,
        "best_hybrid_validation": {"eligible_for_shadow": True},
        "best_hybrid": candidate.__dict__,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, raw = script._load_frozen_candidate(path)
    assert loaded == candidate
    assert raw["holdout_evaluated"] is False

    for key, value, message in (
        ("holdout_evaluated", True, "holdout_evaluated"),
        ("production_promotion_allowed", True, "must not authorize"),
    ):
        invalid = {**manifest, key: value}
        path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            script._load_frozen_candidate(path)
    invalid = {
        **manifest,
        "best_hybrid_validation": {"eligible_for_shadow": False},
    }
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="shadow gate"):
        script._load_frozen_candidate(path)


def test_s10_shadow_cli_dispatch_and_atomic_json(monkeypatch, load_script, tmp_path):
    script = load_script("09_s10_shadow.py")
    output = tmp_path / "nested" / "status.json"
    script._write_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    assert not output.with_suffix(".json.tmp").exists()

    monkeypatch.setattr(script, "freeze", lambda args: {"mode": "freeze"})
    monkeypatch.setattr(script, "operate", lambda args: {"mode": "operate"})
    assert script.run(SimpleNamespace(freeze=True))["mode"] == "freeze"
    assert script.run(SimpleNamespace(freeze=False))["mode"] == "operate"


from vs_epl_krls.audit import append_audit_record, verify_audit_ledger  # noqa: E402


def _parity_forecast_payload(
    *, target: str, point: float, origin_price: float, lower: float, upper: float
) -> dict:
    return {
        "forecast": {
            "origin_date": "2026-08-16",
            "target_date": target,
            "origin_price": origin_price,
            "point": point,
            "lower": lower,
            "upper": upper,
        },
        "decision": {"recommend_prebuy": True},
    }


def test_parity_ledger_settles_only_observed_weeks(tmp_path, load_script):
    """A contagem prospectiva so avanca quando a semana-alvo e realmente
    observada, e cada semana e liquidada uma unica vez."""

    script = load_script("23_s10_parity_production.py")
    ledger = tmp_path / "parity_ledger.jsonl"
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-09", "2026-08-16"]),
            "price": [6.91, 6.89],
        }
    )

    # Sem previsao registrada nao ha nada a liquidar.
    assert script.settle_pending_forecast(ledger, panel) is None

    append_audit_record(
        ledger,
        event="forecast",
        payload=_parity_forecast_payload(
            target="2026-08-23", point=6.9121, origin_price=6.89, lower=6.83, upper=7.01
        ),
    )
    # A semana-alvo ainda nao esta no painel.
    assert script.settle_pending_forecast(ledger, panel) is None
    assert len(verify_audit_ledger(ledger)) == 1


def test_parity_ledger_scores_against_persistence_and_is_idempotent(tmp_path, load_script):
    script = load_script("23_s10_parity_production.py")
    ledger = tmp_path / "parity_ledger.jsonl"
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-16", "2026-08-23"]),
            "price": [6.89, 6.95],
        }
    )
    append_audit_record(
        ledger,
        event="forecast",
        payload=_parity_forecast_payload(
            target="2026-08-23", point=6.9121, origin_price=6.89, lower=6.83, upper=7.01
        ),
    )

    settled = script.settle_pending_forecast(ledger, panel)

    assert settled is not None
    scored = settled["payload"]
    assert scored["target_date"] == "2026-08-23"
    assert scored["observed_price"] == pytest.approx(6.95)
    assert scored["absolute_error"] == pytest.approx(0.0379, abs=1e-6)
    # A persistencia erraria o dobro nesta semana; e a comparacao que importa.
    assert scored["persistence_absolute_error"] == pytest.approx(0.06, abs=1e-6)
    assert scored["interval_covered"] is True
    assert scored["recommended_prebuy"] is True

    # Reexecutar nao pode inflar a contagem prospectiva.
    assert script.settle_pending_forecast(ledger, panel) is None
    records = verify_audit_ledger(ledger)
    assert [record["event"] for record in records] == ["forecast", "realized"]
    assert records[-1]["previous_hash"] == records[0]["record_hash"]


def test_parity_ledger_flags_an_interval_miss(tmp_path, load_script):
    script = load_script("23_s10_parity_production.py")
    ledger = tmp_path / "parity_ledger.jsonl"
    panel = pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-23"]), "price": [7.40]}
    )
    append_audit_record(
        ledger,
        event="forecast",
        payload=_parity_forecast_payload(
            target="2026-08-23", point=6.9121, origin_price=6.89, lower=6.83, upper=7.01
        ),
    )

    scored = script.settle_pending_forecast(ledger, panel)["payload"]

    assert scored["interval_covered"] is False
    assert scored["absolute_error"] == pytest.approx(0.4879, abs=1e-6)
    assert scored["persistence_absolute_error"] == pytest.approx(0.51, abs=1e-6)


def test_parity_evidence_records_a_portable_path(tmp_path, load_script):
    """O manifesto anterior gravava o caminho absoluto da maquina que treinou,
    o que vazava o usuario local e impedia conferir a proveniencia noutro clone."""

    script = load_script("23_s10_parity_production.py")

    inside = script.repo_relative(script.ROOT / "data" / "processed" / "s10_causal_panel.csv")
    assert inside == "data/processed/s10_causal_panel.csv"
    assert ":" not in inside

    outside = script.repo_relative(tmp_path / "painel.csv")
    assert outside.endswith("painel.csv")


def _gate_review_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    dates = pd.date_range("2024-08-18", periods=n, freq="7D")
    actual = 6.0 + np.cumsum(rng.normal(scale=0.01, size=n))
    persistence = np.concatenate([[actual[0]], actual[:-1]])
    paridade = actual + rng.normal(scale=0.02, size=n)
    return pd.DataFrame(
        {
            "target_date": dates,
            "actual": actual,
            "persistence": persistence,
            "paridade": paridade,
            "paridade__lower": paridade - 0.08,
            "paridade__upper": paridade + 0.08,
            "arima": persistence + rng.normal(scale=0.02, size=n),
            "vs_epl_krls": persistence + rng.normal(scale=0.03, size=n),
        }
    )


def test_gate_review_scores_models_without_selecting_any(load_script):
    script = load_script("24_s10_gate_review.py")
    frame = _gate_review_frame()

    verdict = script.review_parity(frame)

    assert verdict["challenger"] == "paridade"
    assert verdict["incumbent"] == "ARIMA"
    assert isinstance(verdict["promote"], bool)
    assert {gate["name"] for gate in verdict["gates"]} >= {
        "mae_melhor_que_incumbente",
        "sem_regressao_em_semana_parada",
        "intervalo_calibrado",
    }


def test_gate_review_compares_the_published_interval_with_the_conformal_one(load_script):
    script = load_script("24_s10_gate_review.py")
    frame = _gate_review_frame(120)

    review = script.review_interval(frame)

    assert review["published"]["n"] == 120
    assert review["adaptive_conformal"]["n"] == 120
    assert len(review["alpha_path"]) == 120
    # A banda publicada e fixa; a conformal tem que variar de largura.
    widths = np.array(review["upper"]) - np.array(review["lower"])
    assert widths.std() > 0


def test_gate_review_weekly_savings_aligns_with_the_metric_series(load_script):
    script = load_script("24_s10_gate_review.py")
    frame = _gate_review_frame()

    savings = script.weekly_savings(frame, "paridade", "paridade")

    assert savings.shape == (len(frame),)
    assert savings[-1] == 0.0  # a ultima semana nao tem decisao a consumir


def test_gate_review_markdown_table_renders_booleans_readably(load_script):
    script = load_script("24_s10_gate_review.py")

    table = script.markdown_table(
        [{"name": "gate", "passed": False, "observed": 0.5}],
        ["name", "passed", "observed"],
    )

    assert "**nao**" in table
    assert "| name | passed | observed |" in table


def test_parity_production_calibrates_the_interval_level(load_script):
    """A escala condicional continua do modelo; o nivel vem do conformal."""

    script = load_script("23_s10_parity_production.py")
    rng = np.random.default_rng(5)
    n = 260
    dates = pd.date_range("2020-01-05", periods=n, freq="7D")
    price = 4.0 + np.cumsum(rng.normal(scale=0.02, size=n))
    causal = pd.DataFrame(
        {
            "date": dates,
            "price": price,
            "brent": 70 + np.cumsum(rng.normal(scale=0.5, size=n)),
            "usdbrl": 5 + np.cumsum(rng.normal(scale=0.01, size=n)),
            "ulsd": 2 + np.cumsum(rng.normal(scale=0.01, size=n)),
            "parity": 3 + np.cumsum(rng.normal(scale=0.01, size=n)),
            "producer_price": 3 + np.cumsum(rng.normal(scale=0.01, size=n)),
        }
    )
    panel = script.build_parity_panel(causal)
    model = script.PassThroughECM(
        config=script.PassThroughConfig(), feature_names=script.PARITY_FEATURES
    ).fit(panel)

    band, diagnostics = script.calibrated_band(model, panel)

    assert diagnostics["available"] is True
    assert band is not None
    assert diagnostics["calibration_weeks"] >= 20
    assert 0.0 < diagnostics["adaptive_conformal"]["empirical_coverage"] <= 1.0
    lower, upper = band.interval(6.0, scale=0.03)
    assert lower < 6.0 < upper


def test_parity_production_declines_to_calibrate_without_history(load_script):
    script = load_script("23_s10_parity_production.py")

    class _Stub:
        config = script.PassThroughConfig()

        def walk_forward(self, panel, start, end):
            return pd.DataFrame(
                {"actual": [1.0], "prediction": [1.0], "sigma": [0.1], "lower": [0.9], "upper": [1.1]}
            )

    band, diagnostics = script.calibrated_band(_Stub(), pd.DataFrame({"a": range(30)}))

    assert band is None
    assert diagnostics["available"] is False


def _spread_panel(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Painel estadual sintetico com spread revertendo a media."""

    rng = np.random.default_rng(seed)
    spread = np.empty(n)
    spread[0] = -0.05
    for index in range(1, n):
        spread[index] = -0.05 + 0.94 * (spread[index - 1] + 0.05) + rng.normal(scale=0.02)
    national = 6.0 + np.cumsum(rng.normal(scale=0.02, size=n))
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-07", periods=n, freq="7D"),
            "price": national + spread,
            "national_price": national,
            "stations": 200,
            "spread": spread,
        }
    )
    frame["origin_price"] = frame["price"].shift(1)
    frame["y"] = frame["spread"].diff()
    frame["spread_lag1"] = frame["spread"].shift(1)
    frame["dspread1"] = frame["y"].shift(1)
    frame["producer_spread_z"] = rng.normal(size=n)
    return frame


def test_horizon_diagnostic_finds_the_signal_where_the_half_life_predicts(load_script):
    """Reversao lenta significa sinal minusculo em uma semana e maior no horizonte."""

    script = load_script("30_s10_vs_on_spread.py")
    panel = _spread_panel(n=500)

    scores = script.horizon_predictability(panel, len(panel))

    assert set(scores) == {"h1", "h2", "h4", "h8", "h12", "h26"}
    # Um spread que reverte devagar e quase imprevisivel em uma semana...
    assert scores["h1"] < 0.15
    # ...e muito mais previsivel em oito e doze.
    assert scores["h12"] > scores["h1"]
    assert scores["h8"] > scores["h2"]


def test_horizon_diagnostic_declines_gracefully_on_a_short_series(load_script):
    script = load_script("30_s10_vs_on_spread.py")

    scores = script.horizon_predictability(_spread_panel(n=40), 40)

    # Horizontes que nao cabem na serie simplesmente nao entram.
    assert "h26" not in scores


def test_vs_on_spread_never_learns_a_week_before_predicting_it(load_script):
    """Prequential estrito: adulterar o futuro nao pode mexer no ja previsto."""

    script = load_script("30_s10_vs_on_spread.py")
    panel = _spread_panel(n=400, seed=5)
    start, end = 300, 340

    baseline = script.vs_walk_forward(panel, start, end)

    tampered = panel.copy()
    tampered.loc[end:, ["spread", "y", "spread_lag1", "dspread1"]] *= 5.0
    after = script.vs_walk_forward(tampered, start, end)

    assert np.allclose(baseline, after, equal_nan=True)
    assert np.isfinite(baseline).sum() > 30


def test_pooled_estimates_only_use_weeks_before_the_fold(load_script):
    script = load_script("28_s10_multi_state.py")
    panels = {uf: _spread_panel(n=400, seed=index) for index, uf in enumerate("ABCD")}
    cut = panels["A"]["date"].iloc[300]

    estimates = script.fold_estimates(panels, cut)

    assert set(estimates) == set("ABCD")
    for kappa, error in estimates.values():
        assert 0.0 <= kappa <= 0.5
        assert error > 0

    # Estragar o futuro nao pode mudar a estimativa do fold.
    tampered = {uf: frame.copy() for uf, frame in panels.items()}
    for frame in tampered.values():
        frame.loc[300:, ["y", "spread_lag1"]] *= 9.0
    assert script.fold_estimates(tampered, cut) == estimates


def test_ledger_review_script_reports_and_signals_failure(load_script, tmp_path, capsys):
    from vs_epl_krls.audit import record_forecast

    script = load_script("29_s10_ledger_review.py")
    ledger = tmp_path / "parado.jsonl"
    record_forecast(
        ledger,
        {
            "artifact_sha256": "a" * 64,
            "forecast": {
                "target_date": "2020-01-05",
                "origin_price": 6.5,
                "point": 6.51,
                "lower": 6.4,
                "upper": 6.6,
            },
            "decision": {"recommend_prebuy": False},
        },
        target_date="2020-01-05",
    )

    import sys as _sys

    argv = _sys.argv
    try:
        _sys.argv = ["29", "--ledger", f"parado={ledger}"]
        code = script.main()
    finally:
        _sys.argv = argv

    # Previsao pendente desde 2020: o processo semanal parou.
    assert code == 1
    assert "liquidacao_atrasada" in capsys.readouterr().out
