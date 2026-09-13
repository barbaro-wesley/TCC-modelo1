from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.experiment import (
    _replay,
    evaluate_fold,
    freeze_experiment,
    load_frozen,
    summarize,
    validate_protocol,
)
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.selection import S10Candidate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def history():
    t = np.arange(146)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-05", periods=len(t), freq="7D"),
            "price": 5.5 + 0.002 * t + 0.03 * np.sin(t / 5),
        }
    )


@pytest.fixture
def protocol(history):
    candidate = S10Candidate(
        "test",
        "price",
        "delta",
        0.05,
        0.03,
        0.94,
        0.74,
        1.0,
        0.5,
        0.001,
        max_dictionary_size=4,
        max_rules=4,
        feature_scaling="robust_bounded",
    )
    return {
        "horizon_weeks": 1,
        "models": ["ARIMA", "persistencia", "VS-ePL-KRLS"],
        "calibration_weeks": 40,
        "calibration_window": 52,
        "allow_anomalous_change": False,
        "candidate": asdict(candidate),
        "development_end": str(history.date.iloc[143].date()),
        "folds": [
            {
                "id": "dev",
                "start": str(history.date.iloc[140].date()),
                "end": str(history.date.iloc[143].date()),
            }
        ],
    }


class FastARIMA:
    def __init__(self, prices):
        self.last = prices[-1]

    def forecast(self, steps=1):
        return np.repeat(self.last, steps)


@pytest.fixture
def fast_arima(monkeypatch):
    # Only replaces costly ARIMA optimization; real preprocessing, VS, Ridge,
    # prediction guardrails, interval adaptation and online learning still run.
    monkeypatch.setattr(S10ProductionForecaster, "_fit_arima", staticmethod(FastARIMA))


def test_freeze_excludes_holdout_and_rejects_overwrite(tmp_path, protocol, history):
    raw = tmp_path / "source.bin"
    raw.write_bytes(b"source snapshot")
    output = tmp_path / "frozen"
    manifest = freeze_experiment(
        ROOT, protocol, history, raw, output, source_commit="test", repository_label="test"
    )
    restored, data = load_frozen(ROOT, output)
    assert restored == manifest
    assert len(data) == 144
    assert data.date.max() == pd.Timestamp(protocol["development_end"])
    with pytest.raises(FileExistsError):
        freeze_experiment(
            ROOT, protocol, history, raw, output, source_commit="test", repository_label="test"
        )


@pytest.mark.parametrize("target", ["manifest.json", "development_history.json"])
def test_tamper_detection(tmp_path, protocol, history, target):
    raw = tmp_path / "raw"
    raw.write_bytes(b"original")
    output = tmp_path / "frozen"
    freeze_experiment(
        ROOT, protocol, history, raw, output, source_commit="test", repository_label="test"
    )
    path = output / target
    if target == "manifest.json":
        value = json.loads(path.read_text(encoding="utf-8"))
        value["source_commit"] = "changed"
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="modified"):
        load_frozen(ROOT, output)


def test_changed_code_or_runtime_rejected(tmp_path, protocol, history, monkeypatch):
    from vs_epl_krls import experiment

    raw = tmp_path / "raw"
    raw.write_bytes(b"original")
    output = tmp_path / "frozen"
    freeze_experiment(
        ROOT, protocol, history, raw, output, source_commit="test", repository_label="test"
    )
    with monkeypatch.context() as patch:
        patch.setattr(experiment, "source_hashes", lambda root: {})
        with pytest.raises(ValueError, match="source changed"):
            load_frozen(ROOT, output)
    monkeypatch.setattr(experiment, "environment", dict)
    with pytest.raises(ValueError, match="runtime differs"):
        load_frozen(ROOT, output)


def test_holdout_and_anomaly_override_are_forbidden(protocol):
    changed = copy.deepcopy(protocol)
    changed["folds"][-1]["end"] = "2024-08-18"
    with pytest.raises(ValueError, match="holdout"):
        validate_protocol(changed)
    protocol["allow_anomalous_change"] = True
    with pytest.raises(ValueError, match="anomaly overrides"):
        validate_protocol(protocol)


def test_production_constructor_preserves_real_forecast(history, protocol):
    # Real statsmodels fit verifies the extracted shared initializer is equivalent.
    candidate = S10Candidate(**protocol["candidate"])
    residuals = np.linspace(-0.07, 0.08, 40)
    old = S10ProductionForecaster(candidate, calibration_residuals=residuals).fit(history)
    old.warm_start_interval_alpha()
    new = S10ProductionForecaster.fit_calibrated(
        candidate, history, calibration_residuals=residuals
    )
    assert old.predict_next().as_dict() == new.predict_next().as_dict()


def test_replay_matches_direct_production_with_fallback(history, protocol, fast_arima):
    candidate = S10Candidate(**protocol["candidate"])
    bundle = S10ProductionForecaster.fit_calibrated(
        candidate,
        history.iloc[:140],
        calibration_residuals=np.linspace(-0.1, 0.1, 40),
        fallback_calibration_residuals=np.linspace(-0.2, 0.2, 40),
    )
    bundle.arima_model_.last = 1000.0  # Force the real production fallback.
    direct = copy.deepcopy(bundle)
    expected = []
    for item in history.iloc[140:142].itertuples(index=False):
        forecast = direct.predict_next()
        expected.append(forecast)
        direct.update_one(item.date, item.price)
    rows = _replay(bundle, history.iloc[140:142], fold_id="dev")
    assert rows[0]["fallback_used"]
    for row, issued in zip(rows, expected):
        assert (row["point"], row["lower"], row["upper"]) == (issued.point, issued.p10, issued.p90)
    assert bundle.predict_next().as_dict() == direct.predict_next().as_dict()
    assert bundle.calibration_residuals.tolist() == direct.calibration_residuals.tolist()
    assert (
        bundle.fallback_calibration_residuals.tolist()
        == direct.fallback_calibration_residuals.tolist()
    )


def test_future_values_do_not_change_prior_forecasts(history, protocol, fast_arima):
    before = history.copy(deep=True)
    first = evaluate_fold(history, protocol, protocol["folds"][0], smoke_weeks=2)
    changed = history.copy()
    changed.loc[142:, "price"] += 50
    second = evaluate_fold(changed, protocol, protocol["folds"][0], smoke_weeks=2)
    assert first == second
    pd.testing.assert_frame_equal(history, before)
    assert len(first) == 6
    assert all(row["calibration_end"] < row["target_date"] for row in first)
    assert all(row["calibration_samples_before_observation"] >= 40 for row in first)
    assert {row["model"] for row in summarize(first)} == set(protocol["models"])


def test_current_label_cannot_affect_issued_prediction(history, protocol, fast_arima):
    first = evaluate_fold(history, protocol, protocol["folds"][0], smoke_weeks=2)
    changed = history.copy()
    changed.loc[140, "price"] += 0.03
    second = evaluate_fold(changed, protocol, protocol["folds"][0], smoke_weeks=2)
    for name in protocol["models"]:
        a = [row for row in first if row["model"] == name]
        b = [row for row in second if row["model"] == name]
        assert (a[0]["point"], a[0]["lower"], a[0]["upper"]) == (
            b[0]["point"],
            b[0]["lower"],
            b[0]["upper"],
        )
        assert a[1]["persistence"] != b[1]["persistence"]


def test_missing_week_rejected_before_fitting(history, protocol, fast_arima):
    with pytest.raises(ValueError, match="missing week"):
        evaluate_fold(history.drop(index=120), protocol, protocol["folds"][0])


def test_freeze_rejects_unusable_calendar_without_writing(tmp_path, history, protocol):
    raw = tmp_path / "raw"
    raw.write_bytes(b"original")
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match="missing week"):
        freeze_experiment(
            ROOT,
            protocol,
            history.drop(index=120),
            raw,
            output,
            source_commit="test",
            repository_label="test",
        )
    assert not output.exists()


def test_short_calibration_keeps_production_nominal_alpha(history, protocol, fast_arima):
    candidate = S10Candidate(**protocol["candidate"])
    bundle = S10ProductionForecaster.fit_calibrated(
        candidate, history, calibration_residuals=np.linspace(-0.1, 0.1, 26)
    )
    assert bundle._interval_alpha() == pytest.approx(0.2)
    assert not getattr(bundle, "interval_alpha_warm_started_", False)
