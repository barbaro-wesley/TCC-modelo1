"""Camada de decisao: recomendacao acionavel, divergencia visivel, governanca explicita."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from vs_epl_krls.anp_official import sha256_file
from vs_epl_krls.decision import S10DecisionService
from vs_epl_krls.product import S10ProductService, compare_runtime_versions
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.selection import S10Candidate


def _candidate() -> S10Candidate:
    return S10Candidate(
        candidate_id="decision-test",
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
def release(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision")
    index = np.arange(120, dtype=float)
    history = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-07", periods=120, freq="7D"),
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
    forecast = model.predict_next()
    return root, artifact, manifest, forecast


def _service(release, *, challenger=None, gates=None) -> S10DecisionService:
    _, artifact, manifest, forecast = release
    target = forecast.target_date
    clock = lambda: datetime.fromisoformat(f"{target}T12:00:00+00:00")  # noqa: E731
    product = S10ProductService(artifact, manifest, clock=clock)
    return S10DecisionService(product, challenger_forecast=challenger, gate_review=gates)


def _write_challenger(root, target_date: str, point: float, name: str = "challenger.json"):
    path = root / name
    path.write_text(
        json.dumps({"forecast": {"target_date": target_date, "point": point}}),
        encoding="utf-8",
    )
    return path


def test_decision_recommends_waiting_when_the_signal_is_below_the_threshold(release) -> None:
    decision = _service(release).decide(200_000)

    assert decision.recommendation == "aguardar"
    assert decision.liters_to_prebuy == 0.0
    assert decision.expected_saving_brl == 0.0
    assert decision.decided_by == decision.views[0].name
    assert "abaixo do limiar" in decision.rationale[0]


def test_decision_sizes_the_prebuy_when_the_signal_clears_the_threshold(release) -> None:
    root, artifact, manifest, forecast = release
    target = forecast.target_date
    clock = lambda: datetime.fromisoformat(f"{target}T12:00:00+00:00")  # noqa: E731
    product = S10ProductService(artifact, manifest, clock=clock)
    # Limiar zerado: qualquer alta prevista dispara, o que permite exercitar o
    # dimensionamento sem depender do numero que o modelo de teste produz.
    service = S10DecisionService(product, signal_threshold=0.0)
    decision = service.decide(200_000)

    if decision.views[0].expected_change_brl_per_liter > 0:
        assert decision.recommendation == "antecipar"
        assert decision.liters_to_prebuy == pytest.approx(200_000 * 12 / 52 * 0.25, rel=1e-6)
        assert decision.expected_saving_brl > 0
    else:
        assert decision.recommendation == "aguardar"


def test_disagreement_is_surfaced_and_lowers_confidence(release) -> None:
    root, _, _, forecast = release
    # Challenger bem acima do preco de origem: direcao oposta a do primario.
    challenger = _write_challenger(root, forecast.target_date, forecast.last_observed_price + 5.0)

    decision = _service(release, challenger=challenger).decide()

    assert decision.models_agree is False
    assert decision.confidence == "baixa"
    assert [view.name for view in decision.views] == [forecast.primary_model, "paridade"]
    assert any("discorda" in reason for reason in decision.rationale)
    # A divergencia nao muda o dono da decisao.
    assert decision.decided_by == forecast.primary_model


def test_challenger_from_a_different_week_is_ignored_instead_of_compared(release) -> None:
    """Comparar semanas diferentes inventaria uma divergencia que nao existe."""

    root, _, _, forecast = release
    stale = _write_challenger(root, "1999-01-03", 99.0, name="stale.json")

    decision = _service(release, challenger=stale).decide()

    assert len(decision.views) == 1
    assert decision.models_agree is True
    assert any("Sem previsao de challenger" in reason for reason in decision.rationale)


def test_agreement_near_the_threshold_is_medium_confidence(release) -> None:
    root, _, _, forecast = release
    aligned = _write_challenger(
        root, forecast.target_date, forecast.last_observed_price, name="aligned.json"
    )

    decision = _service(release, challenger=aligned).decide()

    assert decision.models_agree is True
    assert decision.confidence in {"media", "alta"}


def test_governance_reports_why_the_primary_is_the_primary(release) -> None:
    root, _, _, forecast = release
    gates = root / "gates.json"
    gates.write_text(
        json.dumps(
            {
                "holdout_window": {"start": "2024-08-18", "end": "2026-08-09"},
                "verdict_parity_vs_arima": {
                    "promote": False,
                    "failed_gates": ["mae_melhor_que_incumbente"],
                    "gates": [{"name": "mae_melhor_que_incumbente", "passed": False}],
                },
                "verdict_vs_epl_krls_vs_arima": {
                    "promote": False,
                    "failed_gates": ["economia_supera_incumbente"],
                    "gates": [],
                },
                "interval_review": {"adaptive_conformal": {"empirical_coverage": 0.808}},
            }
        ),
        encoding="utf-8",
    )

    payload = _service(release, gates=gates).governance()

    assert payload["primary_model"] == forecast.primary_model
    verdicts = payload["gate_review"]["verdicts"]
    assert verdicts["paridade_vs_arima"]["promote"] is False
    assert "mae_melhor_que_incumbente" in verdicts["paridade_vs_arima"]["failed_gates"]
    assert payload["gate_review"]["interval_review"]["empirical_coverage"] == 0.808


def test_governance_is_honest_when_there_is_no_gate_review(release) -> None:
    payload = _service(release).governance()

    assert payload["gate_review"] is None
    assert "promocao automatica e proibida" in payload["promotion_policy"]


def test_decision_validates_volume_and_readiness(release) -> None:
    service = _service(release)
    with pytest.raises(ValueError, match="volume_liters"):
        service.decide(0)
    with pytest.raises(ValueError, match="volume_liters"):
        service.decide(60_000_000)

    _, artifact, manifest, _ = release
    expired = S10ProductService(
        artifact, manifest, clock=lambda: datetime(2099, 1, 1, tzinfo=timezone.utc)
    )
    with pytest.raises(RuntimeError, match="nao esta apta a servir"):
        S10DecisionService(expired).decide()


def test_service_rejects_impossible_policy_configuration(release) -> None:
    _, artifact, manifest, _ = release
    product = S10ProductService(artifact, manifest)
    with pytest.raises(ValueError, match="signal_threshold"):
        S10DecisionService(product, signal_threshold=-0.01)
    with pytest.raises(ValueError, match="flexibility"):
        S10DecisionService(product, flexibility=1.5)


def test_runtime_mismatch_is_detected_and_degrades_the_status(release) -> None:
    """O SHA-256 garante os bytes; ele nao garante o runtime que os interpreta."""

    root, artifact, _, forecast = release
    manifest = root / "release_with_runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_sha256": sha256_file(artifact),
                "metadata": {
                    "runtime_versions": {
                        "numpy": "0.0.1",
                        "pandas": "0.0.1",
                        "scikit_learn": "0.0.1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    target = forecast.target_date
    product = S10ProductService(
        artifact,
        manifest,
        clock=lambda: datetime.fromisoformat(f"{target}T12:00:00+00:00"),
    )
    status = product.status()

    assert product.runtime_mismatches
    assert status.runtime_verified is False
    assert status.status == "degraded"
    assert any(reason.startswith("runtime_mismatch:numpy") for reason in status.reasons)
    # Bytes integros: continua servindo, apenas sinalizado.
    assert status.serving_ready is True


def test_runtime_comparison_is_quiet_when_versions_match_or_are_absent() -> None:
    import numpy

    assert compare_runtime_versions(None) == ()
    assert compare_runtime_versions({}) == ()
    assert compare_runtime_versions({"numpy": numpy.__version__}) == ()
    assert compare_runtime_versions({"pacote_desconhecido": "1.0"}) == ()


def _write_regional(root, target_date: str, *, point: float = 6.61, origin: float = 6.58,
                    status: str = "development_only", name: str = "rs.json"):
    path = root / name
    path.write_text(
        json.dumps(
            {
                "uf": "RS",
                "region": "sul",
                "forecast": {
                    "origin_date": "2026-08-16",
                    "target_date": target_date,
                    "origin_price": origin,
                    "national_point": 6.9121,
                    "spread_point": -0.3011,
                    "point": point,
                    "lower": point - 0.05,
                    "upper": point + 0.05,
                    "nominal_coverage": 0.80,
                },
                "decision": {"recommend_prebuy": point - origin > 0.01},
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
                    "status": status,
                    "holdout_read": status == "holdout_read",
                    "prospective_weeks_settled": 0,
                    "prospective_target": 26,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_state_decision_uses_the_price_the_local_buyer_actually_pays(release) -> None:
    root, _, _, forecast = release
    state = _write_regional(root, forecast.target_date)

    service = _service(release)
    service.regional = {"RS": json.loads(state.read_text(encoding="utf-8"))}

    result = service.decide(200_000, uf="RS")

    assert result.scope == "estadual:RS"
    assert result.origin_price_brl_per_liter == pytest.approx(6.58)
    assert result.recommendation == "antecipar"
    assert result.views[0].name == "regional_rs"
    assert any("serie que o comprador do estado enfrenta" in r for r in result.rationale)
    assert any("Decomposicao" in r for r in result.rationale)


def test_development_only_evidence_caps_the_confidence(release) -> None:
    """Um modelo que nunca viu holdout nao pode sair com confianca alta."""

    root, _, _, forecast = release
    service = _service(release)
    # Sinal forte: tres vezes o limiar. Ainda assim nao pode ser "alta".
    service.regional = {
        "RS": json.loads(
            _write_regional(root, forecast.target_date, point=6.61, origin=6.55).read_text(
                encoding="utf-8"
            )
        )
    }

    result = service.decide(200_000, uf="RS")

    assert result.evidence_status == "development_only"
    assert result.confidence == "media"
    assert any("nao sobe acima de media" in r for r in result.rationale)


def test_state_decision_refuses_a_forecast_from_another_week(release) -> None:
    root, _, _, _ = release
    service = _service(release)
    service.regional = {
        "RS": json.loads(
            _write_regional(root, "1999-01-03", name="stale_rs.json").read_text(encoding="utf-8")
        )
    }

    with pytest.raises(RuntimeError, match="aponta para 1999-01-03"):
        service.decide(200_000, uf="RS")


def test_unknown_state_is_a_clear_error_listing_what_exists(release) -> None:
    service = _service(release)

    with pytest.raises(KeyError, match="nenhuma previsao estadual carregada"):
        service.decide(200_000, uf="SP")
    assert service.available_states == ()


def test_basis_report_scales_linearly_with_the_client_volume(release) -> None:
    root, _, _, forecast = release
    service = _service(release)
    service.regional = {
        "RS": json.loads(
            _write_regional(root, forecast.target_date, name="basis_rs.json").read_text(
                encoding="utf-8"
            )
        )
    }

    small = service.basis("rs", 200_000)
    large = service.basis("RS", 1_000_000)

    # 0,14 R$/L * 200.000 * 12 = R$ 336.000/ano
    assert small.annual_budget_error_brl == pytest.approx(336_000.0)
    assert large.annual_budget_error_brl == pytest.approx(5 * small.annual_budget_error_brl)
    # A defasagem corrente e maior que a media: o spread esta em extremo.
    assert small.current_annual_gap_brl == pytest.approx(744_000.0)
    assert small.spread_percentile == pytest.approx(0.006)
    assert "Nao e cotacao de fornecedor" in small.disclaimer


def test_basis_validates_the_volume(release) -> None:
    root, _, _, forecast = release
    service = _service(release)
    service.regional = {
        "RS": json.loads(
            _write_regional(root, forecast.target_date, name="vol_rs.json").read_text(
                encoding="utf-8"
            )
        )
    }

    with pytest.raises(ValueError, match="volume_liters"):
        service.basis("RS", 0)


def test_governance_lists_the_states_available(release) -> None:
    root, _, _, forecast = release
    service = _service(release)
    service.regional = {
        "RS": json.loads(
            _write_regional(root, forecast.target_date, name="gov_rs.json").read_text(
                encoding="utf-8"
            )
        )
    }

    assert service.governance()["states_available"] == ["RS"]
