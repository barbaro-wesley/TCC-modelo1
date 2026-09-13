"""Build, verify and publish the S10-only production forecasting artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vs_epl_krls import load_anp_fuel_csv
from vs_epl_krls.production import S10ProductionForecaster
from vs_epl_krls.selection import S10Candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [
            f"{row[column]:.5f}" if isinstance(row[column], (float, np.floating)) else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _approved_primary(manifest: dict[str, object]) -> str:
    selected = str(manifest["selected_production_model"])
    comparison = {str(row["model"]): row for row in manifest["comparison"]}  # type: ignore[index]
    if selected not in comparison:
        raise RuntimeError("selected model is absent from holdout comparison")
    arima_rmse = float(comparison["ARIMA"]["rmse"])
    selected_rmse = float(comparison[selected]["rmse"])
    if not np.isfinite(selected_rmse) or selected_rmse > 1.02 * arima_rmse:
        return "ARIMA"
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if int(manifest["horizon_weeks"]) != 1:
        raise RuntimeError("production artifact currently supports horizon=1 only")
    candidate = S10Candidate(**manifest["champion_selected_without_holdout"])
    weights = tuple(float(value) for value in manifest["ensemble"]["weights"])
    primary = _approved_primary(manifest)
    residual_table = pd.read_csv(args.calibration_residuals)
    residual_column = {
        "ARIMA": "residual_arima",
        "Ridge": "residual_ridge",
        "persistencia": "residual_persistence",
        "VS-ePL-KRLS": "residual_vs_epl_krls",
        "ensemble": "residual_ensemble",
    }[primary]
    residuals = residual_table[residual_column].to_numpy(float)
    s10 = load_anp_fuel_csv(args.data, products=["S10"], weekly="mean")
    history = s10[["date", "price"]].copy()
    bundle = S10ProductionForecaster(
        candidate,
        primary_model=primary,  # type: ignore[arg-type]
        ensemble_weights=weights,  # type: ignore[arg-type]
        calibration_residuals=residuals,
        fallback_calibration_residuals=residual_table[
            "residual_persistence"
        ].to_numpy(float),
    ).fit(history)
    # Nasce calibrada: sem aquecer, a release comecaria no nivel nominal e levaria
    # dezenas de semanas para descobrir o que a janela de residuos ja diz.
    warm_alpha = bundle.warm_start_interval_alpha()
    artifact = bundle.save(args.artifact)
    before = bundle.predict_next()
    print(
        f"nivel do intervalo aquecido: alpha {warm_alpha:.4f} "
        f"(cobertura nominal implicita {1 - warm_alpha:.1%})"
    )
    load_started = time.perf_counter_ns()
    restored = S10ProductionForecaster.load(artifact)
    load_ms = (time.perf_counter_ns() - load_started) / 1e6
    after = restored.predict_next()
    if before.as_dict() != after.as_dict():
        raise RuntimeError("serialization round-trip changed the forecast")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_hash = _sha256(artifact)
    latencies_ms: list[float] = []
    benchmark_points: list[float] = []
    for _ in range(100):
        prediction_started = time.perf_counter_ns()
        benchmark_points.append(restored.predict_next().point)
        latencies_ms.append((time.perf_counter_ns() - prediction_started) / 1e6)
    runtime_benchmark = {
        "load_ms": float(load_ms),
        "predict_p50_ms": float(np.quantile(latencies_ms, 0.50)),
        "predict_p95_ms": float(np.quantile(latencies_ms, 0.95)),
        "predict_p99_ms": float(np.quantile(latencies_ms, 0.99)),
        "n_predictions": len(latencies_ms),
        "deterministic_points": len(set(benchmark_points)) == 1,
        "artifact_bytes": artifact.stat().st_size,
    }
    payload: dict[str, object] = {
        "forecast": after.as_dict(),
        "health": restored.health().as_dict(),
        "metadata": restored.metadata(),
        "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_hash,
        "runtime_benchmark": runtime_benchmark,
        "selection_manifest": str(args.selection_manifest.resolve()),
        "production_gate": {
            "selection_used_holdout_for_tuning": False,
            "roundtrip_exact": True,
            "primary_approved": primary,
            "vs_epl_krls_promoted": bool(
                manifest["vs_epl_krls_gates"]["promote_vs_epl_krls"]
            ),
        },
    }
    (args.output_dir / "forecast.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    comparison = pd.DataFrame(manifest["comparison"])
    interval = manifest["interval_diagnostics"]
    health = restored.health()
    lines = [
        "# Model card — previsão semanal nacional do Diesel B S10",
        "",
        f"- Modelo primário aprovado: **{primary}**",
        "- Challenger monitorado: VS-ePL-KRLS corrigido e normalizado",
        "- Horizonte: 1 semana",
        "- Fonte: ANP, preço médio nacional de revenda",
        f"- Treino: {restored.training_start_} a {restored.training_end_}",
        f"- Observações: {len(restored.history_)}",
        f"- Fingerprint dos dados: `{restored.data_fingerprint_}`",
        f"- SHA-256 do artefato: `{artifact_hash}`",
        f"- Versão do contrato do artefato: {restored.artifact_version}",
        f"- Latência end-to-end p95 (100 chamadas): {runtime_benchmark['predict_p95_ms']:.2f} ms",
        f"- Tamanho do artefato: {runtime_benchmark['artifact_bytes'] / 1024**2:.2f} MiB",
        "",
        "## Holdout final (104 semanas)",
        "",
        _markdown_table(
            comparison[
                [
                    "model",
                    "rmse",
                    "mae",
                    "smape",
                    "directional_accuracy",
                    "rmse_ratio_vs_naive",
                    "dm_pvalue",
                ]
            ]
        ),
        "",
        "## Intervalo calibrado",
        "",
        f"- Cobertura nominal P10–P90: {100 * float(interval['nominal_coverage']):.1f}%",
        f"- Cobertura no holdout: {100 * float(interval['holdout_coverage']):.1f}%",
        f"- Largura média: R$ {float(interval['mean_width']):.3f}/L",
        f"- Resíduos de calibração: {int(interval['calibration_samples'])}",
        f"- Janela adaptativa máxima após atualização online: {restored.calibration_window} semanas",
        "",
        "## Previsão atual",
        "",
        f"- Última observação: {after.last_observed_date}, R$ {after.last_observed_price:.3f}/L",
        f"- Próxima semana: {after.target_date}",
        f"- Ponto: R$ {after.point:.3f}/L",
        f"- Intervalo P10–P90: R$ {after.p10:.3f}–{after.p90:.3f}/L",
        f"- Fallback usado: {after.fallback_used}",
        "",
        "## Saúde do challenger",
        "",
        f"- Estado: {health.status}",
        f"- Regras: {health.n_rules}/{restored.vs_model_.config.max_rules}",
        f"- Maior dicionário: {health.max_dictionary_size}/{restored.vs_model_.config.max_dictionary_size}",
        f"- Substituições de elementos KRLS: {health.dictionary_replacements} ({100 * health.dictionary_replacement_rate:.2f}% das atualizações)",
        f"- Cobertura online observada: {health.empirical_interval_coverage if health.empirical_interval_coverage is not None else 'aguardando 20 semanas'}",
        f"- MAE online recente: {health.recent_mae if health.recent_mae is not None else 'aguardando observações'}",
        f"- Avisos: {', '.join(health.warnings) if health.warnings else 'nenhum'}",
        "",
        "## Política operacional",
        "",
        "- O VS-ePL-KRLS permanece em shadow mode porque não passou os gates de promoção.",
        "- Previsões não finitas ou mudanças semanais fora do limite robusto acionam fallback.",
        "- Nova semana deve ser incorporada por `update_one`; datas repetidas ou preços inválidos são rejeitados.",
        "- Pressão de regras/dicionários, churn, clipping e cobertura online aparecem em `health()`.",
        "- Reexecutar seleção antes de trocar o primário; nunca promover usando o holdout para ajustar parâmetros.",
        "",
        "## Limitações",
        "",
        "- Preço médio nacional, não preço de um posto ou estado.",
        "- O intervalo por quantis de resíduos é empírico; após cada atualização, usa uma janela móvel e só alerta cobertura após 20 realizações online.",
        "- Choques de política de preços e eventos externos podem exceder os padrões históricos.",
        "- O gap de coleta da ANP em 2020 permanece no histórico e deve ser monitorado.",
    ]
    (args.output_dir / "model_card.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "anp_semanal_desde_2013.xlsx")
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_selection" / "selection_manifest_h1.json",
    )
    parser.add_argument(
        "--calibration-residuals",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_selection" / "calibration_residuals_h1.csv",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "s10_production.joblib",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_production",
    )
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
