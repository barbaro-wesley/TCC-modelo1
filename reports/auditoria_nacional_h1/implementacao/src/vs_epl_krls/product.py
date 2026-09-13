"""Application service and operational contract for the S10 forecast product."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

from .anp_official import sha256_file
from .production import S10Forecast, S10ProductionForecaster


SAO_PAULO = ZoneInfo("America/Sao_Paulo")

#: Pacotes cuja versao muda resultado numerico e, portanto, muda previsao.  O
#: SHA-256 garante que os bytes da release sao os mesmos; ele nao garante que o
#: runtime que le esses bytes e o mesmo que os escreveu.  Sem esta checagem, uma
#: atualizacao de dependencia troca a previsao servida sem nenhum sinal — a
#: mesma classe de falha silenciosa que ja custou uma coluna inteira de dados a
#: este projeto.
NUMERIC_RUNTIME_PACKAGES = ("numpy", "pandas", "scikit_learn", "statsmodels", "joblib")


def _installed_runtime_versions() -> dict[str, str]:
    import joblib
    import numpy
    import pandas
    import sklearn
    import statsmodels

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "joblib": joblib.__version__,
    }


def compare_runtime_versions(recorded: dict[str, Any] | None) -> tuple[str, ...]:
    """Lista os pacotes numericos cuja versao difere da que gerou a release."""

    if not isinstance(recorded, dict):
        return ()
    installed = _installed_runtime_versions()
    mismatches = []
    for package in NUMERIC_RUNTIME_PACKAGES:
        expected = recorded.get(package)
        current = installed.get(package)
        if isinstance(expected, str) and current and expected != current:
            mismatches.append(f"{package}:{expected}!={current}")
    return tuple(mismatches)


def read_trusted_json(path: Path) -> dict[str, Any]:
    """Le um manifesto local rejeitando chave duplicada e token nao finito."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read trusted JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("manifest root must be an object")
    return value


@dataclass(frozen=True)
class S10OperationalStatus:
    status: Literal["healthy", "degraded", "blocked"]
    serving_ready: bool
    integrity_verified: bool
    forecast_fresh: bool
    forecast_target_date: str
    forecast_valid_until: str
    last_observed_date: str
    data_age_days: int
    stale_days: int
    primary_model: str
    challenger_status: str
    runtime_verified: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class S10CostScenario:
    volume_liters: float
    current_unit_price_brl: float
    forecast_unit_price_brl: float
    p10_unit_price_brl: float
    p90_unit_price_brl: float
    current_cost_brl: float
    forecast_cost_brl: float
    p10_cost_brl: float
    p90_cost_brl: float
    expected_change_brl: float
    exposure_per_centavo_brl: float
    disclaimer: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class S10ProductService:
    """Load one immutable, hash-verified model release and expose safe queries."""

    def __init__(
        self,
        artifact: str | Path,
        manifest: str | Path,
        *,
        selection_manifest: str | Path | None = None,
        procurement_report: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.artifact_path = Path(artifact).resolve()
        self.manifest_path = Path(manifest).resolve()
        self.release_manifest = read_trusted_json(self.manifest_path)
        expected_hash = self.release_manifest.get("artifact_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError("release manifest must contain artifact_sha256")
        self.artifact_sha256 = sha256_file(self.artifact_path)
        if self.artifact_sha256.lower() != expected_hash.lower():
            raise RuntimeError("artifact SHA-256 does not match release manifest")
        self.model = S10ProductionForecaster.load(
            self.artifact_path,
            expected_sha256=expected_hash,
        )
        # The serving process is intentionally immutable. Cache values that only
        # change after an offline release update instead of serializing every
        # request through the model's prediction lock.
        self._forecast = self.model.predict_next()
        self._health = self.model.health()
        self.integrity_verified = True
        metadata = self.release_manifest.get("metadata")
        self.runtime_mismatches = compare_runtime_versions(
            metadata.get("runtime_versions") if isinstance(metadata, dict) else None
        )
        self.selection_manifest = (
            read_trusted_json(Path(selection_manifest).resolve())
            if selection_manifest is not None
            else None
        )
        self.procurement_report = (
            read_trusted_json(Path(procurement_report).resolve())
            if procurement_report is not None
            else None
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.started_at_utc = self._clock().astimezone(timezone.utc).isoformat()

    def _local_today(self) -> date:
        observed = self._clock()
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return observed.astimezone(SAO_PAULO).date()

    def status(self) -> S10OperationalStatus:
        forecast = self._forecast
        health = self._health
        target = date.fromisoformat(forecast.target_date)
        valid_until = target.fromordinal(target.toordinal() + 6)
        last_observed = date.fromisoformat(forecast.last_observed_date)
        today = self._local_today()
        stale_days = max(0, (today - valid_until).days)
        reasons: list[str] = []
        if stale_days:
            reasons.append("forecast_expired")
        reasons.extend(f"challenger:{warning}" for warning in health.warnings)
        reasons.extend(f"runtime_mismatch:{item}" for item in self.runtime_mismatches)
        serving_ready = self.integrity_verified and stale_days == 0
        status: Literal["healthy", "degraded", "blocked"]
        if not serving_ready:
            status = "blocked"
        elif health.warnings or self.runtime_mismatches:
            # Degradado, nao bloqueado: os bytes continuam integros e a previsao
            # continua util, mas quem consome precisa saber que o runtime que le
            # a release nao e o que a escreveu.
            status = "degraded"
        else:
            status = "healthy"
        return S10OperationalStatus(
            status=status,
            serving_ready=serving_ready,
            integrity_verified=self.integrity_verified,
            forecast_fresh=stale_days == 0,
            forecast_target_date=forecast.target_date,
            forecast_valid_until=str(valid_until),
            last_observed_date=forecast.last_observed_date,
            data_age_days=max(0, (today - last_observed).days),
            stale_days=stale_days,
            primary_model=forecast.primary_model,
            challenger_status=health.status,
            runtime_verified=not self.runtime_mismatches,
            reasons=tuple(reasons),
        )

    def forecast(self) -> dict[str, object]:
        generated = self._clock()
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
        prediction = self._forecast
        return {
            "contract_version": "1.0",
            "generated_at_utc": generated.astimezone(timezone.utc).isoformat(),
            "forecast": prediction.as_dict(),
            "operational_status": self.status().as_dict(),
            "provenance": {
                "artifact_version": self.model.artifact_version_,
                "artifact_sha256": self.artifact_sha256,
                "data_fingerprint": prediction.data_fingerprint,
                "release_manifest": str(self.manifest_path),
            },
        }

    def cost_scenario(self, volume_liters: float = 200_000.0) -> S10CostScenario:
        volume = float(volume_liters)
        if not 0 < volume <= 50_000_000:
            raise ValueError("volume_liters must be greater than zero and at most 50 million")
        forecast = self._forecast
        return S10CostScenario(
            volume_liters=volume,
            current_unit_price_brl=forecast.last_observed_price,
            forecast_unit_price_brl=forecast.point,
            p10_unit_price_brl=forecast.p10,
            p90_unit_price_brl=forecast.p90,
            current_cost_brl=volume * forecast.last_observed_price,
            forecast_cost_brl=volume * forecast.point,
            p10_cost_brl=volume * forecast.p10,
            p90_cost_brl=volume * forecast.p90,
            expected_change_brl=volume * (forecast.point - forecast.last_observed_price),
            exposure_per_centavo_brl=volume * 0.01,
            disclaimer=(
                "Scenario based on ANP national average resale prices; it is not a supplier "
                "quote, a savings guarantee, or an automatic purchase recommendation."
            ),
        )

    @property
    def current_forecast(self) -> S10Forecast:
        """Return the immutable forecast associated with the loaded release."""

        return self._forecast

    def model_evidence(self) -> dict[str, object]:
        if self.selection_manifest is None:
            return {}
        comparison = self.selection_manifest.get("comparison")
        if not isinstance(comparison, list):
            return {}
        by_model = {
            str(row.get("model")): row
            for row in comparison
            if isinstance(row, dict) and row.get("model")
        }
        arima = by_model.get("ARIMA")
        naive = by_model.get("persistencia")
        if not arima or not naive:
            return {"comparison": comparison}
        arima_rmse = float(arima["rmse"])
        naive_rmse = float(naive["rmse"])
        return {
            "holdout_weeks": int(arima["n"]),
            "primary": "ARIMA",
            "arima_rmse": arima_rmse,
            "arima_mae": float(arima["mae"]),
            "naive_rmse": naive_rmse,
            "rmse_reduction_vs_naive_fraction": 1.0 - arima_rmse / naive_rmse,
            "directional_accuracy": float(arima["directional_accuracy"]),
            "dm_pvalue_vs_naive": float(arima["dm_pvalue"]),
            "statistically_conclusive_at_005": float(arima["dm_pvalue"]) < 0.05,
            "comparison": comparison,
        }

    def model_catalog(self) -> dict[str, object]:
        """Return the models, roles, current outputs and frozen evaluation data."""

        forecast = self._forecast
        evidence = self.model_evidence()
        comparison_value = evidence.get("comparison", [])
        comparison = comparison_value if isinstance(comparison_value, list) else []
        evaluations = {
            str(row["model"]): row
            for row in comparison
            if isinstance(row, dict) and isinstance(row.get("model"), str)
        }
        roles = {
            "ARIMA": ("primary", "active"),
            "persistencia": ("fallback", "available"),
            "VS-ePL-KRLS": ("challenger", "shadow"),
            "ensemble": ("benchmark", "not_promoted"),
            "Ridge": ("benchmark", "not_promoted"),
        }
        names = list(dict.fromkeys([*forecast.components, *evaluations]))
        models: list[dict[str, object]] = []
        for name in names:
            role, lifecycle = roles.get(name, ("benchmark", "not_promoted"))
            model: dict[str, object] = {
                "name": name,
                "role": role,
                "lifecycle": lifecycle,
                "current_prediction_brl_per_liter": forecast.components.get(name),
                "holdout_evaluation": evaluations.get(name),
            }
            if name == "VS-ePL-KRLS":
                model["health"] = self._health.as_dict()
            models.append(model)
        return {
            "contract_version": "1.0",
            "release_artifact_sha256": self.artifact_sha256,
            "forecast_target_date": forecast.target_date,
            "last_observed_date": forecast.last_observed_date,
            "models": models,
            "selection_evidence": evidence,
            "operational_status": self.status().as_dict(),
        }

    def procurement_evidence(self) -> dict[str, object]:
        """Return the governed policy replay, without expanding event details."""

        if self.procurement_report is None:
            return {}
        allowed = (
            "model",
            "period_start",
            "period_end",
            "n_weeks",
            "monthly_liters",
            "flexibility_fraction",
            "signal_threshold_brl_per_liter",
            "carrying_cost_brl_per_liter_week",
            "triggered_prebuys",
            "trigger_precision",
            "net_savings_brl",
            "savings_excluding_largest_event_brl",
            "largest_event_share_of_savings",
            "annualized_savings_brl",
            "annualized_savings_ci90_brl",
            "methodology",
        )
        return {key: self.procurement_report[key] for key in allowed if key in self.procurement_report}
