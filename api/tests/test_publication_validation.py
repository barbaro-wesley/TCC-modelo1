import subprocess
import sys

import pytest

from forecast_store.repository import PublicationError, prepare

from .test_model_publication import payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("previsao_pontual", float("nan")),
        ("previsao_pontual", float("inf")),
        ("previsao_pontual", True),
        ("p10", -1),
        ("p10", None),
        ("p90", 6.0),
        ("semana_prevista", "2000-01-01"),
        ("preco_observado_ultima_semana", 7.0),
    ],
)
def test_rejects_invalid_publication(field, value):
    forecast, series, metrics = payload()
    forecast[field] = value
    with pytest.raises(PublicationError):
        prepare(forecast, series, metrics)


def test_metadata_nan_is_null_and_missing_evaluation_is_rejected():
    forecast, series, metrics = payload()
    metrics[0]["coverage"] = float("nan")
    assert prepare(forecast, series, metrics)[2][0]["values"]["coverage"] is None
    with pytest.raises(PublicationError):
        prepare(forecast, series, [])
    with pytest.raises(PublicationError):
        prepare(forecast, series + series, metrics)


def test_api_does_not_import_training_or_ml():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import api.app.main, sys; "
            "assert not {'training', 'pandas', 'numpy', 'torch', 'pipeline'} & sys.modules.keys()",
        ],
        check=True,
    )
