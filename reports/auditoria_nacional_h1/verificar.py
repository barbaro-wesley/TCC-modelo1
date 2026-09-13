"""Auditoria em leitura do fluxo h=1 de um repositorio externo.

Grava evidencias somente ao lado deste script. Nao executa selecao completa,
downloads, publicacao nem atualizacoes dos artefatos originais.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import warnings

sys.dont_write_bytecode = True
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--commit", required=True, help="Commit conferido externamente com git rev-parse HEAD")
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from vs_epl_krls.fuel import load_anp_fuel_csv
    from vs_epl_krls.selection import (
        S10Candidate, build_s10_feature_frame, build_s10_supervised,
        candidate_grid, evaluate_temporal_fold, pinned_validation_folds,
    )
    from vs_epl_krls.production import S10ProductionForecaster
    from vs_epl_krls.audit import verify_audit_ledger

    def read(rel):
        return json.loads((repo / rel).read_text(encoding="utf-8"))

    tracked = ["src/vs_epl_krls/selection.py", "src/vs_epl_krls/production.py",
               "scripts/05_s10_model_selection.py", "scripts/06_train_s10_production.py",
               "reports/vs_epl_krls/s10_production/forecast.json",
               "reports/vs_epl_krls/s10_selection/selection_manifest_h1.json",
               "data/raw/anp_semanal_desde_2013.xlsx", "artifacts/s10_production.joblib"]
    snapshot = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in tracked}
    commit = args.commit
    evidence = {"repository": str(repo), "commit": commit, "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__}
    raw = repo / "data/raw/anp_semanal_desde_2013.xlsx"
    history = load_anp_fuel_csv(raw, products=["S10"], weekly="mean")
    data = build_s10_supervised(history, horizon=1, feature_set="lags")
    windows = pinned_validation_folds(data.target_dates)
    manifest = read("reports/vs_epl_krls/s10_selection/selection_manifest_h1.json")
    candidate = S10Candidate(**manifest["champion_selected_without_holdout"])
    evidence["data"] = {"n": len(history), "start": str(history.date.min().date()),
                        "end": str(history.date.max().date()),
                        "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                        "windows": windows.as_manifest()}
    days = (data.target_dates - data.dates).astype("timedelta64[D]").astype(int)
    evidence["calendar_mismatches"] = [
        {"origin": str(pd.Timestamp(data.dates[i]).date()),
         "target": str(pd.Timestamp(data.target_dates[i]).date()), "days": int(days[i])}
        for i in np.flatnonzero(days != 7)
    ]
    bad = []
    for t in range(windows.folds[0].validation_start, windows.holdout.validation_end):
        te = data.known_target_end(t)
        if te and np.max(data.target_dates[:te]) > data.dates[t]:
            bad.append(t)
    evidence["future_labels_at_evaluated_origins"] = bad

    feature_results = {}
    cutoff = len(history) - 10
    altered = history.copy()
    altered.loc[cutoff:, "price"] += 100
    for feature_set in ("price", "lags", "dynamics"):
        before = build_s10_feature_frame(history.iloc[:cutoff], feature_set=feature_set)
        after = build_s10_feature_frame(altered, feature_set=feature_set).iloc[:len(before)]
        feature_results[feature_set] = before.reset_index(drop=True).equals(after.reset_index(drop=True))
    evidence["features_unchanged_by_future"] = feature_results

    # Only development is rerun. Saved holdout predictions are read for arithmetic checks.
    fold = windows.folds[-1]
    baseline = evaluate_temporal_fold(candidate, data, fold)
    altered_labels = data.target_price.copy()
    altered_labels[fold.validation_start:] += 50
    changed = evaluate_temporal_fold(candidate, replace(data, target_price=altered_labels), fold)
    saved = pd.read_csv(repo / "reports/vs_epl_krls/s10_selection/holdout_predictions_h1.csv")
    assert np.array_equal(saved.actual.to_numpy(), data.target_price[windows.holdout.validation_start:windows.holdout.validation_end])
    evidence["vs_current_target_perturbation"] = {
        "first_prediction_equal": bool(baseline.predictions[0] == changed.predictions[0]),
        "next_prediction_changes_when_label_arrives": bool(baseline.predictions[1] != changed.predictions[1]),
    }
    evidence["vs_reexecution"] = {
        "metrics": baseline.metrics,
        "evaluation_scope": "last development fold only; legacy candidate parameters with current calendar code",
        "target_start": str(pd.Timestamp(baseline.dates[0]).date()),
        "target_end": str(pd.Timestamp(baseline.dates[-1]).date()),
        "n": len(baseline.predictions),
    }
    evidence["legacy_manifest"] = {
        "pipeline_version": manifest.get("pipeline_version"),
        "has_validation_fingerprint": "validation_fingerprint" in manifest,
        "stored_candidate": asdict(candidate),
        "current_default_scaling": candidate_grid(horizon=1, n_random=0)[0].feature_scaling,
    }

    residuals = pd.read_csv(repo / "reports/vs_epl_krls/s10_selection/calibration_residuals_h1.csv")
    qlo, qhi = np.quantile(residuals.residual_arima, [.1,.9])
    hits = (saved.actual >= saved.arima+qlo)&(saved.actual <= saved.arima+qhi)
    evidence["saved_interval_recalculation"] = {
        "coverage": float(hits.mean()), "width": float(qhi-qlo), "n_calibration": len(residuals),
        "n_test": len(saved), "original_nominal_coverage": .8,
    }

    # Current approve helper genuinely depends on holdout metric values.
    spec = importlib.util.spec_from_file_location("audit_train_script", repo / "scripts/06_train_s10_production.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate_manifest = {"selected_production_model":"ensemble", "comparison":[
        {"model":"ARIMA","rmse":1.}, {"model":"ensemble","rmse":1.01}]}
    first = module._approved_primary(gate_manifest)
    gate_manifest["comparison"][1]["rmse"] = 1.03
    evidence["holdout_changes_approved_model"] = {"rmse_1_01":first,
        "rmse_1_03":module._approved_primary(gate_manifest)}

    # Hash-verified local trusted bundle; changes below only affect an in-memory copy.
    production_manifest = read("reports/vs_epl_krls/s10_production/forecast.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = S10ProductionForecaster.load(repo/"artifacts/s10_production.joblib",
                     expected_sha256=production_manifest["artifact_sha256"])
        forecast = bundle.predict_next()
        evidence["loaded_bundle"] = {
            "stored_artifact_version": bundle.artifact_version_,
            "metadata_reported_version": bundle.metadata()["artifact_version"],
            "stored_forecast":production_manifest["forecast"],
            "loaded_forecast":forecast.as_dict(),
            "exact_forecast_matches": forecast.as_dict()==production_manifest["forecast"],
        }
        gap = copy.deepcopy(bundle)
        gap_date = pd.Timestamp(gap.history_.date.iloc[-1])+pd.Timedelta(days=14)
        gap_error = None
        try:
            gap.update_one(gap_date, float(gap.history_.price.iloc[-1]))
        except ValueError as exc:
            gap_error = str(exc)
        evidence["online_gap_accepted"] = {
            "accepted": gap_error is None, "error": gap_error,
            "forecast_scored_target":forecast.target_date, "observation_date":str(gap_date.date()),
            "interval_samples_added":gap.health().interval_monitor_samples-bundle.health().interval_monitor_samples,
            "warning":list(gap.health().warnings),
        }
        warmed = copy.deepcopy(bundle)
        before = warmed.predict_next()
        alpha = warmed.warm_start_interval_alpha()
        after = warmed.predict_next()
        fixed_low, fixed_high = np.quantile(bundle.calibration_residuals, [.1, .9])
        evidence["interval_rule_comparison"] = {
            "alpha":alpha, "same_point":before.point==after.point,
            "loaded_interval":[before.p10,before.p90],
            "after_repeating_warm_start":[after.p10,after.p90],
            "fixed_10_90_interval_same_point":[before.point+fixed_low,before.point+fixed_high],
            "online_samples":bundle.health().interval_monitor_samples,
        }
    records = verify_audit_ledger(repo/"reports/vs_epl_krls/s10_production/production_ledger.jsonl")
    evidence["production_ledger"] = {
        "valid_hash_chain":True,"records":len(records),
        "events":[{"event":r["event"],"recorded_at_utc":r["recorded_at_utc"],
                   "target_date":r["payload"].get("forecast",r["payload"].get("issued_forecast",{})).get("target_date")}
                  for r in records],
    }
    latest = read("reports/vs_epl_krls/s10_product/latest_release.json")
    evidence["latest_release_pointer"] = {
        "target_date": latest["forecast"]["target_date"],
        "last_observation": latest["forecast"]["last_observed_date"],
        "sha256": latest["artifact_sha256"],
        "current_training_target": forecast.target_date,
        "same_artifact_as_training": latest["artifact_sha256"] == production_manifest["artifact_sha256"],
    }
    raw_cols = pd.read_excel(raw,header=None,nrows=25)
    evidence["raw_input_preview_shape"] = list(raw_cols.shape)
    sources = ["src/vs_epl_krls/selection.py","src/vs_epl_krls/production.py",
               "scripts/05_s10_model_selection.py","scripts/06_train_s10_production.py"]
    evidence["source_sha256"] = {p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sources}
    after = {p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in tracked}
    evidence["snapshot_unchanged_during_run"] = snapshot == after
    evidence["snapshot_sha256"] = snapshot
    assert snapshot == after, "Repository changed during audit; repeat against stable inputs"
    path = Path(__file__).with_name("evidencias.json")
    path.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding="utf-8")
    print(json.dumps(evidence,indent=2,ensure_ascii=False,default=str,allow_nan=False))
    print("Evidencias:",path)


if __name__ == "__main__":
    main()
