"""Leak-safe textual news-pressure benchmark on frozen S10 development folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.classical import arima_forecast
from vs_epl_krls.hybrid import HybridFoldResult, evaluate_residual_hybrid
from vs_epl_krls.news_pressure import (
    NEWS_PRESSURE_FEATURES,
    NewsPressureConfig,
    NewsPressureFeatures,
    augment_supervised_with_pressure,
    build_weekly_news_documents,
    generate_prequential_pressure_features,
    load_news_corpus,
    market_pressure_labels,
)
from vs_epl_krls.selection import (
    S10Candidate,
    S10Supervised,
    TemporalFold,
    build_s10_supervised,
)

LATE_FUSION_FEATURES = (
    "news_pressure_down_probability",
    "news_pressure_neutral_probability",
    "news_pressure_up_probability",
    "news_pressure_score",
    "news_pressure_confidence",
    "news_pressure_entropy",
    "news_pressure_no_news",
    "news_pressure_article_count_log1p",
)


@dataclass(frozen=True)
class TextCandidate:
    candidate_id: str
    lookback_days: int
    min_relevance_s10: float


TEXT_CANDIDATES = (
    TextCandidate("pressure_domain_7d", 7, 0.01),
    TextCandidate("pressure_domain_28d", 28, 0.01),
    TextCandidate("pressure_all_7d", 7, 0.0),
    TextCandidate("pressure_all_28d", 28, 0.0),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"data", "revenda"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"S10 panel is missing columns: {sorted(missing)}")
    output = frame.rename(columns={"data": "date", "revenda": "price"})
    output["date"] = pd.to_datetime(output["date"], errors="raise")
    output["price"] = pd.to_numeric(output["price"], errors="raise")
    output = output.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(output) < 200 or not np.all(np.isfinite(output["price"])):
        raise ValueError("S10 panel is too short or contains non-finite prices")
    return output


def _development_panel(
    panel: pd.DataFrame,
    *,
    holdout_start: int,
    common_warmup: int = 12,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Cut before any holdout target; index mapping follows build_s10_feature_frame."""

    cutoff_index = common_warmup + holdout_start
    if cutoff_index >= len(panel):
        raise ValueError("holdout boundary lies outside the S10 panel")
    cutoff = pd.Timestamp(panel.iloc[cutoff_index]["date"])
    return panel.loc[panel["date"] <= cutoff].copy(), cutoff


def _exact_week_samples(data: S10Supervised) -> tuple[S10Supervised, list[str]]:
    day_delta = (data.target_dates - data.dates).astype("timedelta64[D]").astype(int)
    keep = day_delta == 7 * data.horizon
    removed = [
        f"{pd.Timestamp(origin).date()}->{pd.Timestamp(target).date()}"
        for origin, target in zip(data.dates[~keep], data.target_dates[~keep], strict=True)
    ]
    return (
        S10Supervised(
            x=data.x[keep].copy(),
            target_price=data.target_price[keep].copy(),
            origin_price=data.origin_price[keep].copy(),
            dates=data.dates[keep].copy(),
            target_dates=data.target_dates[keep].copy(),
            feature_names=data.feature_names,
            horizon=data.horizon,
        ),
        removed,
    )


def _remap_fold_by_target_dates(
    original_data: S10Supervised,
    corrected_data: S10Supervised,
    fold: TemporalFold,
) -> TemporalFold:
    first_target = original_data.target_dates[fold.validation_start]
    last_target = original_data.target_dates[fold.validation_end - 1]
    indices = np.flatnonzero(
        (corrected_data.target_dates >= first_target)
        & (corrected_data.target_dates <= last_target)
    )
    if not len(indices) or not np.array_equal(
        indices, np.arange(indices[0], indices[-1] + 1)
    ):
        raise RuntimeError(f"cannot remap temporal fold {fold.fold_id} contiguously")
    return TemporalFold(fold.fold_id, int(indices[0]), int(indices[-1] + 1))


def _causal_arima_predictions(
    data: S10Supervised,
    *,
    min_history: int = 52,
    refit_every: int = 13,
) -> np.ndarray:
    predictions = np.full(data.n_samples, np.nan, dtype=float)
    model = None
    last_fit = -10**9
    for origin in range(min_history, data.n_samples):
        history = data.origin_price[: origin + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model is None or origin - last_fit >= refit_every:
                forecast, model = arima_forecast(history, steps=1, model=None)
                last_fit = origin
            else:
                forecast, model = arima_forecast(history, steps=1, model=model)
        predictions[origin] = float(forecast[0])
    return predictions


def _candidate_results(
    candidate: S10Candidate,
    data: S10Supervised,
    folds: list[TemporalFold],
    arima: np.ndarray,
) -> list[HybridFoldResult]:
    return [evaluate_residual_hybrid(candidate, data, fold, arima) for fold in folds]


def _ranking(
    results: dict[str, list[HybridFoldResult]],
    baseline: list[HybridFoldResult],
) -> pd.DataFrame:
    baseline_by_fold = {result.fold_id: result for result in baseline}
    rows: list[dict[str, object]] = []
    for candidate_id, candidate_results in results.items():
        ratios_current = [
            result.metrics["rmse"] / baseline_by_fold[result.fold_id].metrics["rmse"]
            for result in candidate_results
        ]
        ratios_arima = [result.rmse_ratio_vs_base for result in candidate_results]
        replacement_rate = max(
            result.dictionary_replacement_rate for result in candidate_results
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "mean_rmse": float(
                np.mean([result.metrics["rmse"] for result in candidate_results])
            ),
            "worst_rmse": float(
                np.max([result.metrics["rmse"] for result in candidate_results])
            ),
            "mean_mae": float(
                np.mean([result.metrics["mae"] for result in candidate_results])
            ),
            "mean_smape": float(
                np.mean([result.metrics["smape"] for result in candidate_results])
            ),
            "mean_rmse_ratio_vs_current": float(np.mean(ratios_current)),
            "worst_rmse_ratio_vs_current": float(np.max(ratios_current)),
            "mean_rmse_ratio_vs_arima": float(np.mean(ratios_arima)),
            "worst_rmse_ratio_vs_arima": float(np.max(ratios_arima)),
            "replacement_rate": float(replacement_rate),
            "max_rules": max(
                int(np.max(result.rule_counts)) for result in candidate_results
            ),
            "max_dictionary_size": max(
                int(np.max(result.dictionary_sizes)) for result in candidate_results
            ),
            "latency_ms_p95": max(
                result.correction_latency_ms_p95 for result in candidate_results
            ),
        }
        row["beats_current_all_folds"] = max(ratios_current) < 1.0
        row["beats_arima_all_folds"] = max(ratios_arima) < 1.0
        row["bounded_replacement_churn"] = replacement_rate <= 0.40
        row["eligible_for_future_shadow"] = bool(
            row["beats_current_all_folds"]
            and row["beats_arima_all_folds"]
            and row["bounded_replacement_churn"]
        )
        row["selection_score"] = float(
            np.mean(ratios_current)
            + 0.2 * np.max(ratios_current)
            + 0.1 * np.std(ratios_current, ddof=1)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["eligible_for_future_shadow", "selection_score", "mean_rmse"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _classification_metrics(
    pressure: NewsPressureFeatures,
    fold: TemporalFold,
) -> dict[str, object]:
    selected = pressure.frame.iloc[fold.validation_start : fold.validation_end]
    actual = selected["pressure_true_class"].to_numpy(dtype=int)
    predicted = selected["pressure_predicted_class"].to_numpy(dtype=int)
    probabilities = selected[
        [
            "news_pressure_down_probability",
            "news_pressure_neutral_probability",
            "news_pressure_up_probability",
        ]
    ].to_numpy(dtype=float)
    class_to_column = {-1: 0, 0: 1, 1: 2}
    actual_columns = np.asarray([class_to_column[label] for label in actual])
    one_hot = np.eye(3)[actual_columns]
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in (-1, 0, 1):
        true_positive = int(np.sum((actual == label) & (predicted == label)))
        false_positive = int(np.sum((actual != label) & (predicted == label)))
        false_negative = int(np.sum((actual == label) & (predicted != label)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        recalls.append(recall)
        f1_scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    confidence = np.max(probabilities, axis=1)
    correct = (actual == predicted).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return {
        "fold_id": fold.fold_id,
        "n": len(selected),
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(
            -np.mean(np.log(np.clip(probabilities[np.arange(len(actual)), actual_columns], 1e-12, 1.0)))
        ),
        "expected_calibration_error": float(ece),
        "no_news_rate": float(selected["news_pressure_no_news"].mean()),
        "mean_training_samples": float(selected["pressure_training_samples"].mean()),
    }


def _prediction_rows(
    results: dict[str, list[HybridFoldResult]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate_id, candidate_results in results.items():
        for result in candidate_results:
            for date, actual, predicted, base, correction in zip(
                result.dates,
                result.actual,
                result.predictions,
                result.base_predictions,
                result.corrections,
                strict=True,
            ):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "fold_id": result.fold_id,
                        "target_date": pd.Timestamp(date),
                        "actual": float(actual),
                        "prediction": float(predicted),
                        "arima_prediction": float(base),
                        "residual_correction": float(correction),
                    }
                )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            values.append(f"{float(value):.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _plot(
    output_dir: Path,
    ranking: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    ordered = ranking.sort_values("mean_rmse_ratio_vs_current", ascending=False)
    axes[0].barh(ordered["candidate_id"], ordered["mean_rmse_ratio_vs_current"])
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Impacto no forecast")
    axes[0].set_xlabel("RMSE relativo ao híbrido atual")
    summary = classification.groupby("candidate_id", as_index=False)[
        "balanced_accuracy"
    ].mean()
    axes[1].barh(summary["candidate_id"], summary["balanced_accuracy"])
    axes[1].axvline(1 / 3, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Classificação temporal de pressão")
    axes[1].set_xlabel("Acurácia balanceada")
    figure.tight_layout()
    figure.savefig(output_dir / "news_pressure_backtest.png", dpi=170)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    next_manifest = json.loads(args.next_manifest.read_text(encoding="utf-8"))
    previous_news_manifest = json.loads(
        args.previous_news_manifest.read_text(encoding="utf-8")
    )
    if previous_news_manifest.get("holdout_evaluated") is not False:
        raise RuntimeError("previous news experiment does not prove holdout isolation")
    original_folds = [
        TemporalFold(**value) for value in selection_manifest["validation_folds"]
    ]
    holdout = TemporalFold(**selection_manifest["holdout"])
    if max(fold.validation_end for fold in original_folds) != holdout.validation_start:
        raise RuntimeError("development folds do not end exactly before the frozen holdout")

    panel = _load_panel(args.panel)
    development_panel, cutoff = _development_panel(
        panel, holdout_start=holdout.validation_start
    )
    row_shift_development = build_s10_supervised(
        development_panel, horizon=1, feature_set="dynamics"
    )
    if row_shift_development.n_samples != holdout.validation_start:
        raise RuntimeError("development cutoff does not reproduce the frozen boundary")
    base_data, removed_nonweekly_pairs = _exact_week_samples(row_shift_development)
    folds = [
        _remap_fold_by_target_dates(row_shift_development, base_data, fold)
        for fold in original_folds
    ]
    if max(fold.validation_end for fold in folds) != base_data.n_samples:
        raise RuntimeError("corrected folds do not end at the development boundary")

    corpus = load_news_corpus(args.news_records)
    expected_news_hash = previous_news_manifest.get("parent_news_dataset_sha256")
    if corpus.dataset_sha256 != expected_news_hash:
        raise RuntimeError("raw news corpus differs from the previously audited lineage")
    classifier_config = NewsPressureConfig(
        neutral_threshold=args.neutral_threshold,
        n_features=args.text_features,
        min_training_samples=args.min_training_samples,
        probability_shrinkage=args.probability_shrinkage,
        random_state=args.random_state,
    )

    started = time.perf_counter()
    pressure_by_candidate: dict[str, NewsPressureFeatures] = {}
    datasets: dict[str, S10Supervised] = {}
    document_quality: dict[str, dict[str, object]] = {}
    for candidate in TEXT_CANDIDATES:
        documents = build_weekly_news_documents(
            corpus,
            base_data.dates,
            lookback_days=candidate.lookback_days,
            min_relevance_s10=candidate.min_relevance_s10,
        )
        pressure = generate_prequential_pressure_features(
            base_data, documents, config=classifier_config
        )
        pressure_by_candidate[candidate.candidate_id] = pressure
        datasets[candidate.candidate_id] = augment_supervised_with_pressure(
            base_data, pressure, feature_names=LATE_FUSION_FEATURES
        )
        validation = pressure.frame.iloc[folds[0].validation_start :]
        document_quality[candidate.candidate_id] = {
            "lookback_days": candidate.lookback_days,
            "min_relevance_s10": candidate.min_relevance_s10,
            "development_no_news_rate": float(
                pressure.frame["news_pressure_no_news"].mean()
            ),
            "validation_no_news_rate": float(validation["news_pressure_no_news"].mean()),
            "validation_mean_article_count_log1p": float(
                validation["news_pressure_article_count_log1p"].mean()
            ),
        }

    arima = _causal_arima_predictions(base_data)
    current = S10Candidate(**next_manifest["best_hybrid"])
    baseline_candidate = replace(current, candidate_id="current_no_text_pressure")
    baseline_results = _candidate_results(
        baseline_candidate, base_data, folds, arima
    )
    results: dict[str, list[HybridFoldResult]] = {
        "current_no_text_pressure": baseline_results
    }
    for candidate in TEXT_CANDIDATES:
        price_candidate = replace(current, candidate_id=candidate.candidate_id)
        results[candidate.candidate_id] = _candidate_results(
            price_candidate,
            datasets[candidate.candidate_id],
            folds,
            arima,
        )
    elapsed = time.perf_counter() - started

    ranking = _ranking(results, baseline_results)
    classification = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                **_classification_metrics(
                    pressure_by_candidate[candidate.candidate_id], fold
                ),
            }
            for candidate in TEXT_CANDIDATES
            for fold in folds
        ]
    )
    predictions = _prediction_rows(results)
    pressure_validation = pd.concat(
        [
            pressure_by_candidate[candidate.candidate_id].frame.iloc[
                folds[0].validation_start :
            ].assign(candidate_id=candidate.candidate_id)
            for candidate in TEXT_CANDIDATES
        ],
        ignore_index=True,
    )
    eligible = ranking[ranking["eligible_for_future_shadow"]]
    selected = None if eligible.empty else str(eligible.iloc[0]["candidate_id"])
    best_classifier = str(
        classification.groupby("candidate_id")["balanced_accuracy"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    validation_indices = np.concatenate(
        [np.arange(fold.validation_start, fold.validation_end) for fold in folds]
    )
    validation_labels = market_pressure_labels(
        base_data, neutral_threshold=classifier_config.neutral_threshold
    )[validation_indices]
    label_counts = {
        str(label): int(np.sum(validation_labels == label)) for label in (-1, 0, 1)
    }
    majority_oracle_accuracy = max(label_counts.values()) / len(validation_labels)
    best_classifier_rows = classification[
        classification["candidate_id"] == best_classifier
    ]
    classifier_research_gate = {
        "balanced_accuracy_above_constant_class_each_fold": bool(
            (best_classifier_rows["balanced_accuracy"] > 1 / 3).all()
        ),
        "mean_accuracy_above_validation_majority_oracle": bool(
            best_classifier_rows["accuracy"].mean() > majority_oracle_accuracy
        ),
        "mean_macro_f1_above_one_third": bool(
            best_classifier_rows["macro_f1"].mean() > 1 / 3
        ),
        "mean_expected_calibration_error_below_025": bool(
            best_classifier_rows["expected_calibration_error"].mean() < 0.25
        ),
    }
    classifier_research_gate["passed"] = all(classifier_research_gate.values())

    ranking.to_csv(args.output_dir / "ranking.csv", index=False)
    classification.to_csv(args.output_dir / "classification.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    pressure_validation.to_csv(
        args.output_dir / "pressure_validation_predictions.csv", index=False
    )
    fold_rows = pd.DataFrame(
        [
            {"candidate_id": candidate_id, **result.summary_row()}
            for candidate_id, candidate_results in results.items()
            for result in candidate_results
        ]
    )
    fold_rows.to_csv(args.output_dir / "folds.csv", index=False)
    _plot(args.output_dir, ranking, classification)

    payload: dict[str, object] = {
        "scope": "Diesel S10 textual pressure challenger — frozen development only",
        "label_method": "future_s10_delta_weak_supervision",
        "labels_are_human_annotations": False,
        "holdout_evaluated": False,
        "production_promotion_allowed": False,
        "development_cutoff": cutoff.date().isoformat(),
        "development_samples_after_gap_correction": base_data.n_samples,
        "folds": [asdict(fold) for fold in folds],
        "original_folds": [asdict(fold) for fold in original_folds],
        "removed_nonweekly_pairs": removed_nonweekly_pairs,
        "news_dataset_sha256": corpus.dataset_sha256,
        "parent_provenance_dataset_sha256": corpus.parent_provenance_dataset_sha256,
        "source_catalog_sha256": corpus.source_catalog_sha256,
        "panel_sha256": _sha256_file(args.panel),
        "selection_manifest_sha256": _sha256_file(args.selection_manifest),
        "next_manifest_sha256": _sha256_file(args.next_manifest),
        "previous_news_manifest_sha256": _sha256_file(args.previous_news_manifest),
        "classifier_config": asdict(classifier_config),
        "text_candidates": [asdict(candidate) for candidate in TEXT_CANDIDATES],
        "feature_names": list(NEWS_PRESSURE_FEATURES),
        "document_quality": document_quality,
        "elapsed_seconds": elapsed,
        "selected_for_future_shadow": selected,
        "best_classifier_for_annotation_research": best_classifier,
        "classification_reference": {
            "validation_label_counts": label_counts,
            "validation_majority_oracle_accuracy": majority_oracle_accuracy,
            "constant_class_balanced_accuracy": 1 / 3,
        },
        "classifier_research_gate": classifier_research_gate,
        "selection_gate": (
            "beats current and ARIMA in every development fold, with dictionary "
            "replacement rate <= 0.40"
        ),
        "ranking": ranking.to_dict(orient="records"),
        "classification_mean": classification.groupby("candidate_id", as_index=False)
        .mean(numeric_only=True)
        .to_dict(orient="records"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    report_ranking = ranking[
        [
            "candidate_id",
            "mean_rmse",
            "mean_rmse_ratio_vs_current",
            "worst_rmse_ratio_vs_current",
            "mean_mae",
            "replacement_rate",
            "eligible_for_future_shadow",
        ]
    ]
    report_classification = (
        classification.groupby("candidate_id", as_index=False)[
            [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "multiclass_brier",
                "log_loss",
                "expected_calibration_error",
            ]
        ]
        .mean()
        .sort_values("balanced_accuracy", ascending=False)
    )
    report = [
        "# Backtest de pressão textual — Diesel S10",
        "",
        "O holdout final não foi avaliado. Os rótulos são supervisão fraca baseada no movimento posterior do preço, não anotações humanas e não evidência causal.",
        "",
        f"- Dataset de notícias: `{corpus.dataset_sha256}`",
        f"- Corte de desenvolvimento: `{cutoff.date().isoformat()}`",
        f"- Limiar neutro: `R$ {classifier_config.neutral_threshold:.3f}/L`",
        f"- Candidato apto para shadow futuro: `{selected}`",
        f"- Melhor classificador para pesquisa de anotação: `{best_classifier}`",
        f"- Gate de pesquisa do classificador: `{classifier_research_gate['passed']}`",
        f"- Tempo total: `{elapsed:.2f} s`",
        "",
        "## Impacto na previsão",
        "",
        _markdown_table(report_ranking),
        "",
        "## Qualidade da classificação prequential",
        "",
        _markdown_table(report_classification),
        "",
        f"Referências: maioria do próprio período = `{majority_oracle_accuracy:.3f}` de acurácia; classe constante = `0.333` de acurácia balanceada. Essas referências não são modelos de produção.",
        "",
        "## Decisão",
        "",
        "A promoção continua proibida. Um sinal textual só pode seguir para shadow se passar o gate em todos os folds; rótulos humanos independentes e semanas futuras ainda são necessários para validação profissional.",
    ]
    (args.output_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=ROOT / "data" / "processed" / "semanal_s10_features.csv",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_selection"
        / "selection_manifest_h1.json",
    )
    parser.add_argument(
        "--next-manifest",
        type=Path,
        default=ROOT
        / "reports"
        / "vs_epl_krls"
        / "s10_next"
        / "next_challenger_manifest.json",
    )
    parser.add_argument(
        "--previous-news-manifest",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_news" / "manifest.json",
    )
    parser.add_argument("--news-records", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_news_pressure",
    )
    parser.add_argument("--neutral-threshold", type=float, default=0.010)
    parser.add_argument("--text-features", type=int, default=2**14)
    parser.add_argument("--min-training-samples", type=int, default=52)
    parser.add_argument("--probability-shrinkage", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=20260823)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
