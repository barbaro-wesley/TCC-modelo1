from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vs_epl_krls.anp_official import sha256_file
from vs_epl_krls.api import APISettings, create_app
from vs_epl_krls.product import S10ProductService
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.selection import S10Candidate


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    candidate = S10Candidate(
        candidate_id="api-test", feature_set="price", target_mode="delta",
        alpha=0.05, beta_initial=0.03, alpha_vs1=0.94, alpha_vs2=0.74,
        error_threshold=1.0, kernel_sigma=0.5, regularization=1e-3,
        novelty_factor=0.2, max_dictionary_size=4,
        enable_rule_merging=False, adapt_kernel_width=False,
    )
    index = np.arange(85, dtype=float)
    model = S10ProductionForecaster(
        candidate, calibration_residuals=np.linspace(-0.08, 0.08, 30)
    ).fit(pd.DataFrame({"date": pd.date_range("2024-01-07", periods=85, freq="7D"), "price": 6 + index * 0.002}))
    artifact = model.save(root / "api.joblib")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"artifact_sha256": sha256_file(artifact)}), encoding="utf-8")
    target = model.predict_next().target_date
    def clock():
        return datetime.fromisoformat(target).replace(tzinfo=timezone.utc)

    return S10ProductService(artifact, manifest, clock=clock)


def test_api_auth_schema_security_headers_and_metrics(service):
    app = create_app(
        service,
        settings=APISettings(environment="test", api_key="correct", rate_limit_per_minute=20),
    )
    client = TestClient(app)
    assert client.get("/v1/health/live").status_code == 200
    assert client.get("/v1/health/ready").status_code == 200
    assert client.get("/v1/forecast").status_code == 401
    response = client.get("/v1/forecast", headers={"X-API-Key": "correct"})
    assert response.status_code == 200
    assert response.json()["contract_version"] == "1.0"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-security-policy"] == (
        "default-src 'none'; frame-ancestors 'none'"
    )
    models = client.get("/v1/models", headers={"X-API-Key": "correct"})
    assert models.status_code == 200
    assert any(model["role"] == "primary" for model in models.json()["models"])
    scenario = client.post(
        "/v1/scenarios/cost",
        headers={"X-API-Key": "correct"},
        json={"volume_liters": 200_000},
    )
    assert scenario.status_code == 200
    assert scenario.json()["exposure_per_centavo_brl"] == 2_000
    assert "s10_product_ready 1" in client.get("/metrics").text
    root = client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("application/json")
    assert root.json()["resources"]["models"] == "/v1/models"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").headers["content-type"].startswith(
        "application/json"
    )


def test_api_rejects_unknown_fields_large_bodies_and_rate_abuse(service):
    client = TestClient(
        create_app(
            service,
            settings=APISettings(
                environment="test", api_key="key", rate_limit_per_minute=1,
                max_request_bytes=1024,
            ),
        )
    )
    headers = {"X-API-Key": "key"}
    assert client.get("/v1/forecast", headers=headers).status_code == 200
    assert client.get("/v1/forecast", headers=headers).status_code == 429
    oversized = client.post(
        "/v1/scenarios/cost",
        headers={**headers, "Content-Length": "2048"},
        content=b"{}",
    )
    assert oversized.status_code == 413


def test_production_requires_api_key(service):
    with pytest.raises(RuntimeError, match="mandatory"):
        create_app(service, settings=APISettings(environment="production"))
    production = TestClient(
        create_app(
            service,
            settings=APISettings(environment="production", api_key="secret"),
        )
    )
    assert production.get("/docs").status_code == 404
    assert production.get("/openapi.json").status_code == 200


def test_settings_are_strictly_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("S10_ENVIRONMENT", "production")
    monkeypatch.setenv("S10_API_KEY", "secret")
    monkeypatch.setenv("S10_RATE_LIMIT_PER_MINUTE", "321")
    monkeypatch.setenv("S10_MAX_REQUEST_BYTES", "4096")
    monkeypatch.setenv("S10_ALLOWED_HOSTS", "api.example.test, localhost")
    settings = APISettings.from_environment()
    assert settings.environment == "production"
    assert settings.api_key == "secret"
    assert settings.rate_limit_per_minute == 321
    assert settings.max_request_bytes == 4096
    assert settings.allowed_hosts == ("api.example.test", "localhost")


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("S10_ENVIRONMENT", "preview", "development, test, or production"),
        ("S10_RATE_LIMIT_PER_MINUTE", "0", "between 1 and 10000"),
        ("S10_MAX_REQUEST_BYTES", "100", "between 1024 and 1048576"),
    ],
)
def test_settings_reject_invalid_environment_values(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        APISettings.from_environment()


def test_api_blocks_stale_release_and_normalizes_bad_request_id(service):
    original_clock = service._clock
    stale_time = datetime.fromisoformat(service.current_forecast.target_date).replace(
        tzinfo=timezone.utc
    ) + timedelta(days=8)
    service._clock = lambda: stale_time
    try:
        client = TestClient(
            create_app(
                service,
                settings=APISettings(environment="test", api_key="key"),
            )
        )
        headers = {"X-API-Key": "key", "X-Request-ID": "invalid id!"}
        assert client.get("/v1/health/ready").status_code == 503
        forecast = client.get("/v1/forecast", headers=headers)
        assert forecast.status_code == 503
        assert forecast.headers["X-Request-ID"] != "invalid id!"
        assert client.post(
            "/v1/scenarios/cost",
            headers=headers,
            json={"volume_liters": 200_000},
        ).status_code == 503
        assert client.get("/v1/evidence", headers=headers).status_code == 200
        assert client.get("/v1/models", headers=headers).status_code == 503
    finally:
        service._clock = original_clock


def test_decision_endpoints_require_configuration(service):
    """Sem camada de decisao configurada o recurso responde 404, nao 500."""

    app = create_app(service, settings=APISettings(environment="test"))
    client = TestClient(app)

    assert client.post("/v1/decision", json={"volume_liters": 200_000}).status_code == 404
    assert client.get("/v1/governance").status_code == 404
    assert "decision" in client.get("/").json()["resources"]


def test_decision_endpoint_returns_an_actionable_recommendation(service, tmp_path):
    from vs_epl_krls.decision import S10DecisionService

    challenger = tmp_path / "challenger.json"
    forecast = service.current_forecast
    challenger.write_text(
        json.dumps(
            {
                "forecast": {
                    "target_date": forecast.target_date,
                    "point": forecast.last_observed_price + 5.0,
                }
            }
        ),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps(
            {
                "holdout_window": {"start": "2024-08-18", "end": "2026-08-09"},
                "verdict_parity_vs_arima": {"promote": False, "failed_gates": ["x"], "gates": []},
            }
        ),
        encoding="utf-8",
    )
    app = create_app(
        service,
        settings=APISettings(environment="test"),
        decision_service=S10DecisionService(
            service, challenger_forecast=challenger, gate_review=gates
        ),
    )
    client = TestClient(app)

    response = client.post("/v1/decision", json={"volume_liters": 200_000})
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendation"] in {"antecipar", "aguardar"}
    assert payload["models_agree"] is False
    assert payload["confidence"] == "baixa"
    assert len(payload["views"]) == 2
    assert payload["disclaimer"]

    governance = client.get("/v1/governance")
    assert governance.status_code == 200
    assert governance.json()["gate_review"]["verdicts"]["paridade_vs_arima"]["promote"] is False

    # Contrato fechado: campo desconhecido e rejeitado como nos demais recursos.
    assert client.post("/v1/decision", json={"volume": 10}).status_code == 422


def _regional_payload(target_date: str) -> dict:
    return {
        "uf": "RS",
        "region": "sul",
        "forecast": {
            "origin_date": "2026-08-16",
            "target_date": target_date,
            "origin_price": 6.58,
            "national_point": 6.9121,
            "spread_point": -0.3011,
            "point": 6.611,
            "lower": 6.5583,
            "upper": 6.6607,
            "nominal_coverage": 0.80,
        },
        "decision": {"recommend_prebuy": True},
        "basis": {
            "uf": "RS",
            "as_of": "2026-08-16",
            "state_price_brl_per_liter": 6.58,
            "national_price_brl_per_liter": 6.89,
            "current_spread_brl_per_liter": -0.31,
            "relative_difference": -0.045,
            "mean_absolute_spread_brl_per_liter": 0.14,
            "spread_z": -2.96,
            "spread_percentile": 0.006,
            "history_weeks": 702,
            "window_weeks": 52,
        },
        "evidence": {
            "status": "development_only",
            "holdout_read": False,
            "prospective_weeks_settled": 0,
            "prospective_target": 26,
        },
    }


def _regional_app(service, tmp_path):
    from vs_epl_krls.decision import S10DecisionService

    path = tmp_path / "rs.json"
    path.write_text(
        json.dumps(_regional_payload(service.current_forecast.target_date)), encoding="utf-8"
    )
    return create_app(
        service,
        settings=APISettings(environment="test"),
        decision_service=S10DecisionService(service, regional_forecasts={"RS": path}),
    )


def test_decision_endpoint_serves_the_state_the_buyer_actually_pays(service, tmp_path):
    client = TestClient(_regional_app(service, tmp_path))

    response = client.post("/v1/decision", json={"volume_liters": 200_000, "uf": "RS"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["scope"] == "estadual:RS"
    assert payload["evidence_status"] == "development_only"
    assert payload["confidence"] in {"baixa", "media"}
    assert payload["origin_price_brl_per_liter"] == pytest.approx(6.58)
    assert "basis" in client.get("/").json()["resources"]


def test_basis_endpoint_scales_the_budget_error_by_volume(service, tmp_path):
    client = TestClient(_regional_app(service, tmp_path))

    small = client.post("/v1/basis", json={"uf": "RS", "volume_liters": 200_000}).json()
    large = client.post("/v1/basis", json={"uf": "RS", "volume_liters": 1_000_000}).json()

    assert small["annual_budget_error_brl"] == pytest.approx(336_000.0)
    assert large["annual_budget_error_brl"] == pytest.approx(1_680_000.0)
    assert small["uf"] == "RS"


def test_state_endpoints_reject_unknown_and_malformed_states(service, tmp_path):
    client = TestClient(_regional_app(service, tmp_path))

    assert client.post("/v1/basis", json={"uf": "SP"}).status_code == 404
    assert client.post("/v1/decision", json={"uf": "SP"}).status_code == 404
    # Contrato fechado: UF fora do formato de duas letras nao chega ao servico.
    assert client.post("/v1/basis", json={"uf": "RIO"}).status_code == 422
    assert client.post("/v1/decision", json={"uf": "1"}).status_code == 422


def test_basis_is_unavailable_without_the_decision_layer(service):
    client = TestClient(create_app(service, settings=APISettings(environment="test")))

    assert client.post("/v1/basis", json={"uf": "RS"}).status_code == 404
