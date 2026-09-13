"""Frozen, development-only evaluation through the real production bundle."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .metrics import regression_report
from .production import S10ProductionForecaster
from .selection import S10_HOLDOUT_START, S10Candidate

MODELS = ("ARIMA", "persistencia", "VS-ePL-KRLS")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        **{
            package: version(package)
            for package in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "joblib")
        },
    }


def source_hashes(root: Path) -> dict:
    paths = sorted((root / "src" / "vs_epl_krls").rglob("*.py"))
    paths += [
        root / "scripts" / name
        for name in ("06_train_s10_production.py", "34_s10_frozen_experiment.py")
    ]
    return {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in paths}


def validate_protocol(protocol: dict) -> None:
    if protocol["horizon_weeks"] != 1 or tuple(protocol["models"]) != MODELS:
        raise ValueError("protocol requires national h=1 and the three frozen models")
    if protocol["calibration_weeks"] < 20:
        raise ValueError("at least 20 prior calibration weeks are required")
    if protocol["calibration_window"] < protocol["calibration_weeks"]:
        raise ValueError("calibration window must contain the calibration period")
    if protocol["allow_anomalous_change"] is not False:
        raise ValueError("automatic anomaly overrides are forbidden in this protocol")
    candidate = S10Candidate(**protocol["candidate"])
    if candidate.feature_set not in {"price", "lags", "dynamics"}:
        raise ValueError("only endogenous features belong to the frozen experiment")
    folds = protocol["folds"]
    if not folds or len({fold["id"] for fold in folds}) != len(folds):
        raise ValueError("fold identifiers must be nonempty and unique")
    previous_end = None
    for fold in folds:
        start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        if start > end or (end - start).days % 7 or start.dayofweek != 6:
            raise ValueError("fold dates must form a Sunday weekly calendar")
        if end >= pd.Timestamp(S10_HOLDOUT_START):
            raise ValueError("the already-used holdout cannot enter development")
        if previous_end is not None and start <= previous_end:
            raise ValueError("development folds must be ordered and non-overlapping")
        previous_end = end
    if pd.Timestamp(protocol["development_end"]) != previous_end:
        raise ValueError("development_end must equal the last fold end")


def _check_calendar(history: pd.DataFrame, protocol: dict, start, end) -> None:
    calibration_start = start - pd.Timedelta(weeks=protocol["calibration_weeks"])
    expected = pd.date_range(calibration_start - pd.Timedelta(weeks=13), end, freq="7D")
    if not expected.isin(history.date).all():
        raise ValueError("missing week in calibration/evaluation or feature warmup")
    if (history.date < calibration_start).sum() < 80:
        raise ValueError("at least 80 observations must precede calibration")


def freeze_experiment(
    root: Path,
    protocol: dict,
    history: pd.DataFrame,
    raw_path: Path,
    output: Path,
    *,
    source_commit: str,
    repository_label: str,
) -> dict:
    """Write a new snapshot; never replace an existing experiment directory."""
    validate_protocol(protocol)
    history = S10ProductionForecaster._validate_history(history)
    development = history.loc[
        history.date <= pd.Timestamp(protocol["development_end"]), ["date", "price"]
    ].copy()
    if development.empty or development.date.iloc[-1] != pd.Timestamp(protocol["development_end"]):
        raise ValueError("data do not cover the frozen development endpoint")
    for fold in protocol["folds"]:
        _check_calendar(
            development, protocol, pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
        )
    development["date"] = development.date.dt.strftime("%Y-%m-%d")
    snapshot = encoded(development.to_dict(orient="records"))
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository_label,
        "source_commit": source_commit,
        "source_hashes": source_hashes(root),
        "environment": environment(),
        "raw_source": {"path": str(raw_path), "sha256": digest(raw_path.read_bytes())},
        "development_sha256": digest(snapshot),
        "development_rows": len(development),
        "protocol": protocol,
        "status": "frozen_development_protocol_not_preregistered_before_legacy_holdout",
    }
    manifest["experiment_id"] = digest(encoded(manifest))
    output.mkdir(parents=True, exist_ok=False)
    (output / "development_history.json").write_bytes(snapshot)
    (output / "manifest.json").write_bytes(encoded(manifest))
    return manifest


def load_frozen(root: Path, frozen: Path) -> tuple[dict, pd.DataFrame]:
    manifest = json.loads((frozen / "manifest.json").read_text(encoding="utf-8"))
    identity = dict(manifest)
    expected = identity.pop("experiment_id")
    if digest(encoded(identity)) != expected:
        raise ValueError("frozen manifest was modified")
    validate_protocol(manifest["protocol"])
    if manifest["source_hashes"] != source_hashes(root):
        raise ValueError("source changed; create an explicitly versioned new experiment")
    if manifest["environment"] != environment():
        raise ValueError("runtime differs from the frozen environment")
    snapshot = (frozen / "development_history.json").read_bytes()
    if digest(snapshot) != manifest["development_sha256"]:
        raise ValueError("development snapshot was modified")
    history = pd.DataFrame(json.loads(snapshot))
    history["date"] = pd.to_datetime(history.date)
    history = S10ProductionForecaster._validate_history(history)
    if len(history) != manifest["development_rows"] or (
        history.date.max() > pd.Timestamp(manifest["protocol"]["development_end"])
    ):
        raise ValueError("snapshot escapes the frozen development scope")
    return manifest, history


def _replay(bundle, observations: pd.DataFrame, *, fold_id: str) -> list[dict]:
    """Capture issued forecast first, then score and update with its observation."""
    rows = []
    for observation in observations.itertuples(index=False):
        issued = bundle.predict_next()
        if pd.Timestamp(issued.target_date) != observation.date:
            raise ValueError("missing week: observation does not match the issued forecast")
        alpha = bundle._interval_alpha()
        residual_count = len(bundle.calibration_residuals)
        row = {
            "fold": fold_id,
            "model": issued.primary_model,
            "origin_date": issued.last_observed_date,
            "target_date": issued.target_date,
            "point": issued.point,
            "lower": issued.p10,
            "upper": issued.p90,
            "actual": float(observation.price),
            "alpha_before_observation": alpha,
            "calibration_samples_before_observation": residual_count,
            "fallback_used": issued.fallback_used,
            "fallback_reason": issued.fallback_reason,
            "raw_component": issued.components[issued.primary_model],
            "persistence": issued.components["persistencia"],
            "history_fingerprint_before_observation": issued.data_fingerprint,
        }
        # Uses the production source/anomaly/cadence protections without overrides.
        bundle.update_one(observation.date, float(observation.price))
        rows.append(row)
    return rows


def evaluate_fold(
    history: pd.DataFrame,
    protocol: dict,
    fold: dict,
    *,
    smoke_weeks: int | None = None,
) -> list[dict]:
    validate_protocol(protocol)
    if fold not in protocol["folds"]:
        raise ValueError("fold is not part of the frozen protocol")
    if smoke_weeks is not None and smoke_weeks < 1:
        raise ValueError("smoke_weeks must be positive")
    start, end = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    if smoke_weeks is not None:
        end = min(end, start + pd.Timedelta(weeks=smoke_weeks - 1))
    calibration_start = start - pd.Timedelta(weeks=protocol["calibration_weeks"])
    # Truncate before any fitting or feature construction, including future test data.
    history = history.loc[history.date <= end, ["date", "price"]].copy()
    history = S10ProductionForecaster._validate_history(history)
    _check_calendar(history, protocol, start, end)
    calibration = history.loc[(history.date >= calibration_start) & (history.date < start)]
    evaluation = history.loc[(history.date >= start) & (history.date <= end)]
    initial = history.loc[history.date < calibration_start]
    fit_history = history.loc[history.date < start]
    candidate = S10Candidate(**protocol["candidate"])
    output = []
    for name in protocol["models"]:
        print(f"{fold['id']} / {name}: calibracao causal e replay", flush=True)
        pilot = S10ProductionForecaster.fit_calibrated(
            candidate,
            initial,
            primary_model=name,
            calibration_window=protocol["calibration_window"],
        )
        prior = _replay(pilot, calibration, fold_id="calibration")
        # Match production: primary and persistence calibration are separate streams.
        primary_residuals = [row["actual"] - row["raw_component"] for row in prior]
        fallback_residuals = [row["actual"] - row["persistence"] for row in prior]
        bundle = S10ProductionForecaster.fit_calibrated(
            candidate,
            fit_history,
            primary_model=name,
            calibration_residuals=primary_residuals,
            fallback_calibration_residuals=fallback_residuals,
            calibration_window=protocol["calibration_window"],
        )
        rows = _replay(bundle, evaluation, fold_id=fold["id"])
        for row in rows:
            row["calibration_start"] = str(calibration_start.date())
            row["calibration_end"] = str((start - pd.Timedelta(weeks=1)).date())
        output.extend(rows)
    return output


def summarize(rows: list[dict]) -> list[dict]:
    table = pd.DataFrame(rows)
    results = []
    for (fold, model), group in table.groupby(["fold", "model"], sort=False):
        actual, prediction = group.actual.to_numpy(), group.point.to_numpy()
        results.append(
            {
                "fold": fold,
                "model": model,
                "n": len(group),
                **regression_report(actual, prediction),
                "coverage": float(
                    ((group.lower <= group.actual) & (group.actual <= group.upper)).mean()
                ),
                "mean_width": float((group.upper - group.lower).mean()),
                "fallback_count": int(group.fallback_used.sum()),
            }
        )
    return results
