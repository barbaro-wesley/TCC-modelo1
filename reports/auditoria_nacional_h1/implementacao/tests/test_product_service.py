from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.anp_official import sha256_file
from vs_epl_krls.product import S10ProductService
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.selection import S10Candidate


def _candidate() -> S10Candidate:
    return S10Candidate(
        candidate_id="service-test",
        feature_set="price",
        target_mode="delta",
        alpha=0.05,
        beta_initial=0.03,
        alpha_vs1=0.94,
        alpha_vs2=0.74,
        error_threshold=1.0,
        kernel_sigma=0.5,
        regularization=1e-3,
        novelty_factor=0.2,
        max_dictionary_size=4,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )


@pytest.fixture(scope="module")
def service_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("product-service")
    index = np.arange(100, dtype=float)
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=100, freq="7D"),
            "price": 6 + 0.002 * index + 0.03 * np.sin(index / 8),
        }
    )
    model = S10ProductionForecaster(
        _candidate(), calibration_residuals=np.linspace(-0.1, 0.1, 40)
    ).fit(history)
    artifact = model.save(root / "release.joblib")
    manifest = root / "release.json"
    manifest.write_text(
        json.dumps({"artifact_sha256": sha256_file(artifact)}), encoding="utf-8"
    )
    selection = root / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "comparison": [
                    {"model": "ARIMA", "rmse": 0.08, "mae": 0.03, "n": 104, "directional_accuracy": 0.60, "dm_pvalue": 0.13},
                    {"model": "persistencia", "rmse": 0.10, "mae": 0.04, "n": 104, "directional_accuracy": 0.0, "dm_pvalue": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    target = model.predict_next().target_date
    return artifact, manifest, selection, target


def _clock(day: str):
    return lambda: (
        datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        + pd.Timedelta(hours=12)
    )


def test_product_service_verifies_release_and_reports_freshness(service_files):
    artifact, manifest, selection, target = service_files
    service = S10ProductService(
        artifact, manifest, selection_manifest=selection, clock=_clock(target)
    )
    status = service.status()
    assert status.serving_ready
    assert status.integrity_verified
    assert service.forecast()["provenance"]["artifact_sha256"] == sha256_file(artifact)
    evidence = service.model_evidence()
    assert evidence["rmse_reduction_vs_naive_fraction"] == pytest.approx(0.20)
    assert evidence["statistically_conclusive_at_005"] is False


def test_product_service_blocks_expired_forecast(service_files):
    artifact, manifest, selection, target = service_files
    stale_day = str(pd.Timestamp(target) + pd.Timedelta(days=7))
    service = S10ProductService(
        artifact, manifest, selection_manifest=selection, clock=_clock(stale_day)
    )
    assert service.status().status == "blocked"
    assert service.status().stale_days == 1
    assert "forecast_expired" in service.status().reasons


def test_cost_scenario_and_model_catalog_are_bounded_and_honest(service_files):
    artifact, manifest, selection, target = service_files
    service = S10ProductService(
        artifact, manifest, selection_manifest=selection, clock=_clock(target)
    )
    scenario = service.cost_scenario(200_000)
    assert scenario.exposure_per_centavo_brl == 2_000
    assert scenario.p10_cost_brl <= scenario.forecast_cost_brl <= scenario.p90_cost_brl
    catalog = service.model_catalog()
    assert catalog["contract_version"] == "1.0"
    models = {model["name"]: model for model in catalog["models"]}
    assert models["ARIMA"]["role"] == "primary"
    assert models["VS-ePL-KRLS"]["lifecycle"] == "shadow"
    assert models["persistencia"]["role"] == "fallback"
    with pytest.raises(ValueError, match="volume_liters"):
        service.cost_scenario(0)


def test_product_service_rejects_manifest_hash_mismatch(service_files, tmp_path):
    artifact, _, _, _ = service_files
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"artifact_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not match"):
        S10ProductService(artifact, manifest)
