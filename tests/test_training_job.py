"""Synthetic orchestration checks; never downloads data or fits expensive models."""

import sys

import pandas as pd
import pytest

from data import build, download
from eval.temporal import TEMPORAL_PROTOCOL
from pipeline import forecast, weekly


class RecordingPublisher:
    def __init__(self, state=None):
        self.saved_state = state or {}
        self.events = []

    def start(self, **kwargs):
        self.events.append(("start", kwargs))
        return 42

    def state(self):
        return self.saved_state

    def finish(self, run_id, status, **kwargs):
        self.events.append((status, run_id, kwargs))

    def publish(self, run_id, prediction, series, metrics, **kwargs):
        self.events.append(("published", run_id, prediction, series, metrics))
        return "published"


@pytest.fixture
def job(monkeypatch, tmp_path):
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=400, freq="7D")
    series = pd.DataFrame({"data": dates, "revenda": 6.8})
    prediction = {
        "modelo": "test", "ultima_semana_observada": dates[-1].date().isoformat(),
        "semana_prevista": (dates[-1] + pd.Timedelta(days=7)).date().isoformat(),
        "temporal_protocol": TEMPORAL_PROTOCOL,
    }
    steps = []
    monkeypatch.setattr(weekly, "RAW", tmp_path)
    monkeypatch.setattr(weekly, "RES", tmp_path)
    monkeypatch.setattr(weekly, "LOCK", tmp_path / ".pipeline.lock")
    monkeypatch.setattr(download, "download_all", lambda **kw: steps.append("download"))
    monkeypatch.setattr(build, "load_weekly_s10", lambda: series)
    monkeypatch.setattr(build, "save_processed", lambda: steps.append("build"))
    monkeypatch.setattr(weekly, "rodar_script", lambda *a: steps.append(a))
    monkeypatch.setattr(forecast, "montar_previsao", lambda: prediction)
    pd.DataFrame([{"model": "test", "horizon": 1, "rmse": 0.1}]).to_csv(
        tmp_path / "semanal_benchmarks.csv", index=False
    )
    return steps, prediction, tmp_path


def test_job_publishes_once_without_json_contract(job):
    steps, prediction, directory = job
    pub = RecordingPublisher()
    result = weekly.executar_com_trava(publisher=pub, pular_lstm=True)
    assert result["status"] == "atualizado"
    assert steps == ["download", "build", ("03_semanal.py", {"SKIP_LSTM": "1"})]
    assert [event[0] for event in pub.events] == ["start", "published"]
    assert len(pub.events[-1][3]) == 400
    assert pub.events[-1][2] == prediction
    assert not list(directory.glob("*.json"))


@pytest.mark.parametrize("data_only", [False, True])
def test_skip_and_data_only_never_publish(job, data_only):
    steps, prediction, _ = job
    pub = RecordingPublisher({
        "ultima_semana_processada": prediction["ultima_semana_observada"],
        "temporal_protocol": TEMPORAL_PROTOCOL,
    })
    weekly.executar(publisher=pub, somente_dados=data_only)
    assert pub.events[-1][0] == ("data_only" if data_only else "no_change")
    assert steps == (["download", "build"] if data_only else ["download"])


def test_changed_protocol_retrains_same_week(job):
    _, prediction, _ = job
    pub = RecordingPublisher({"ultima_semana_processada": prediction["ultima_semana_observada"]})
    assert weekly.executar(publisher=pub)["status"] == "atualizado"


def test_failure_records_safe_error_and_does_not_publish(job, monkeypatch):
    pub = RecordingPublisher()

    def fail(*args):
        raise RuntimeError("private details must stay out of public status")

    monkeypatch.setattr(weekly, "rodar_script", fail)
    with pytest.raises(RuntimeError):
        weekly.executar(publisher=pub)
    assert [e[0] for e in pub.events] == ["start", "failed"]
    assert pub.events[-1][2]["error_code"] == "RuntimeError"


def test_rejects_truncated_input_even_when_week_unchanged(job):
    steps, prediction, _ = job
    pub = RecordingPublisher({
        "ultima_semana_processada": prediction["ultima_semana_observada"],
        "temporal_protocol": TEMPORAL_PROTOCOL, "n_linhas_semanal": 401,
    })
    with pytest.raises(weekly.DadosSuspeitos):
        weekly.executar(publisher=pub)
    assert steps == ["download"]
    assert pub.events[-1][0] == "failed"


def test_local_lock_prevents_concurrent_workdir_writes(job):
    from filelock import FileLock

    with FileLock(str(weekly.LOCK)):
        with pytest.raises(weekly.PipelineOcupado):
            weekly.executar_com_trava(publisher=RecordingPublisher())


def test_training_does_not_import_api(job):
    assert not any(name == "api" or name.startswith("api.") for name in sys.modules)
