"""Leak-safe DieselNews feature backtest on frozen S10 development folds only."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, replace
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
from vs_epl_krls.news import (
    NEWS_ALL_FEATURES,
    NEWS_CORE_FEATURES,
    augment_supervised_with_news,
    load_weekly_news_features,
)
from vs_epl_krls.selection import (
    S10Candidate,
    S10Supervised,
    TemporalFold,
    build_s10_supervised,
)

IMPACT_FEATURES = (
    "news_relevance",
    "news_intensity",
    "news_confidence",
    "news_direction_balance",
)


def _load_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"data", "revenda"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"S10 panel is missing columns: {sorted(missing)}")
    return frame.rename(columns={"data": "date", "revenda": "price"})


def _causal_arima_predictions(
    data: S10Supervised,
    *,
    end: int,
    min_history: int = 52,
    refit_every: int = 13,
) -> np.ndarray:
    predictions = np.full(data.n_samples, np.nan, dtype=float)
    model = None
    last_fit = -10**9
    for origin in range(min_history, end):
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


def _exact_week_samples(data: S10Supervised) -> tuple[S10Supervised, list[str]]:
    """Remove row-shift pairs whose target is not exactly one calendar week later."""

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
    return TemporalFold(
        fold_id=fold.fold_id,
        validation_start=int(indices[0]),
        validation_end=int(indices[-1] + 1),
    )


def _candidate_results(
    candidate: S10Candidate,
    data: S10Supervised,
    folds: list[TemporalFold],
    base_predictions: np.ndarray,
) -> list[HybridFoldResult]:
    return [
        evaluate_residual_hybrid(candidate, data, fold, base_predictions) for fold in folds
    ]


def _ranking(
    candidates: dict[str, list[HybridFoldResult]],
    baseline: list[HybridFoldResult],
) -> pd.DataFrame:
    baseline_by_fold = {result.fold_id: result for result in baseline}
    rows: list[dict[str, object]] = []
    for candidate_id, results in candidates.items():
        ratios_current = [
            result.metrics["rmse"] / baseline_by_fold[result.fold_id].metrics["rmse"]
            for result in results
        ]
        ratios_arima = [result.rmse_ratio_vs_base for result in results]
        replacement_rate = max(result.dictionary_replacement_rate for result in results)
        row = {
            "candidate_id": candidate_id,
            "mean_rmse": float(np.mean([result.metrics["rmse"] for result in results])),
            "worst_rmse": float(np.max([result.metrics["rmse"] for result in results])),
            "mean_mae": float(np.mean([result.metrics["mae"] for result in results])),
            "mean_smape": float(np.mean([result.metrics["smape"] for result in results])),
            "mean_rmse_ratio_vs_current": float(np.mean(ratios_current)),
            "worst_rmse_ratio_vs_current": float(np.max(ratios_current)),
            "mean_rmse_ratio_vs_arima": float(np.mean(ratios_arima)),
            "worst_rmse_ratio_vs_arima": float(np.max(ratios_arima)),
            "replacement_rate": replacement_rate,
            "max_rules": max(int(np.max(result.rule_counts)) for result in results),
            "max_dictionary_size": max(
                int(np.max(result.dictionary_sizes)) for result in results
            ),
            "latency_ms_p95": max(result.correction_latency_ms_p95 for result in results),
        }
        row["beats_current_all_folds"] = row["worst_rmse_ratio_vs_current"] < 1.0
        row["beats_arima_all_folds"] = row["worst_rmse_ratio_vs_arima"] < 1.0
        row["bounded_replacement_churn"] = replacement_rate <= 0.40
        row["eligible_for_future_shadow"] = bool(
            row["beats_current_all_folds"]
            and row["beats_arima_all_folds"]
            and row["bounded_replacement_churn"]
        )
        row["selection_score"] = (
            float(row["mean_rmse_ratio_vs_current"])
            + 0.2 * float(row["worst_rmse_ratio_vs_current"])
            + 0.1 * float(np.std(ratios_current, ddof=1))
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["eligible_for_future_shadow", "selection_score", "mean_rmse"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def _block_bootstrap(
    candidate: list[HybridFoldResult],
    baseline: list[HybridFoldResult],
    *,
    repeats: int = 5000,
    block_length: int = 8,
    random_state: int = 20260817,
) -> dict[str, float]:
    candidate_loss = np.concatenate(
        [(result.actual - result.predictions) ** 2 for result in candidate]
    )
    baseline_loss = np.concatenate(
        [(result.actual - result.predictions) ** 2 for result in baseline]
    )
    differential = candidate_loss - baseline_loss
    rng = np.random.default_rng(random_state)
    n = len(differential)
    starts = np.arange(max(1, n - block_length + 1))
    estimates = np.empty(repeats, dtype=float)
    blocks_needed = int(np.ceil(n / block_length))
    for index in range(repeats):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [differential[start : start + block_length] for start in sampled_starts]
        )[:n]
        estimates[index] = float(np.mean(sample))
    return {
        "mean_mse_difference_candidate_minus_current": float(np.mean(differential)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(estimates < 0.0)),
        "n_predictions": n,
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
                        "actual": actual,
                        "prediction": predicted,
                        "arima_prediction": base,
                        "residual_correction": correction,
                    }
                )
    return pd.DataFrame(rows)


def _plot(output_dir: Path, ranking: pd.DataFrame, predictions: pd.DataFrame, winner: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    ordered = ranking.sort_values("mean_rmse_ratio_vs_current", ascending=False)
    axes[0].barh(ordered["candidate_id"], ordered["mean_rmse_ratio_vs_current"])
    axes[0].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("RMSE relativo ao híbrido atual")
    axes[0].set_title("Folds de desenvolvimento")
    current = predictions[predictions["candidate_id"] == "current_no_news"]
    selected = predictions[predictions["candidate_id"] == winner]
    axes[1].plot(current["target_date"], current["actual"], label="real", color="black")
    axes[1].plot(
        current["target_date"], current["prediction"], label="atual sem notícias", alpha=0.8
    )
    axes[1].plot(
        selected["target_date"], selected["prediction"], label=winner, alpha=0.8
    )
    axes[1].set_title("Previsões sequenciais")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "news_backtest.png", dpi=170)
    plt.close(figure)


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
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_panel(args.panel)
    selection_manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    next_manifest = json.loads(args.next_manifest.read_text(encoding="utf-8"))
    original_folds = [
        TemporalFold(**value) for value in selection_manifest["validation_folds"]
    ]
    original_holdout = TemporalFold(**selection_manifest["holdout"])
    if max(fold.validation_end for fold in original_folds) != original_holdout.validation_start:
        raise RuntimeError("development folds do not end exactly before the frozen holdout")
    row_shift_data = build_s10_supervised(panel, horizon=1, feature_set="dynamics")
    base_data, removed_nonweekly_pairs = _exact_week_samples(row_shift_data)
    folds = [
        _remap_fold_by_target_dates(row_shift_data, base_data, fold)
        for fold in original_folds
    ]
    holdout = _remap_fold_by_target_dates(row_shift_data, base_data, original_holdout)
    development_end = max(fold.validation_end for fold in folds)
    if development_end != holdout.validation_start:
        raise RuntimeError("corrected development folds do not end before the frozen holdout")
    news_snapshot = load_weekly_news_features(args.news_signals, horizon_weeks=1)
    datasets = {
        "news_impact": augment_supervised_with_news(
            base_data, news_snapshot, feature_names=IMPACT_FEATURES
        ),
        "news_core": augment_supervised_with_news(
            base_data, news_snapshot, feature_names=NEWS_CORE_FEATURES
        ),
        "news_all": augment_supervised_with_news(
            base_data, news_snapshot, feature_names=NEWS_ALL_FEATURES
        ),
    }
    if any(data.n_samples != base_data.n_samples for data in datasets.values()):
        raise RuntimeError("news augmentation changed supervised fold alignment")

    current = S10Candidate(**next_manifest["best_hybrid"])
    candidates: dict[str, tuple[S10Candidate, S10Supervised]] = {
        "news_impact_sigma015": (
            replace(current, candidate_id="news_impact_sigma015"),
            datasets["news_impact"],
        ),
        "news_core_sigma015": (
            replace(current, candidate_id="news_core_sigma015"),
            datasets["news_core"],
        ),
        "news_core_sigma030": (
            replace(current, candidate_id="news_core_sigma030", kernel_sigma=0.30),
            datasets["news_core"],
        ),
        "news_all_sigma030": (
            replace(current, candidate_id="news_all_sigma030", kernel_sigma=0.30),
            datasets["news_all"],
        ),
    }
    started = time.perf_counter()
    arima = _causal_arima_predictions(base_data, end=development_end)
    baseline_candidate = replace(current, candidate_id="current_no_news")
    baseline_results = _candidate_results(
        baseline_candidate, base_data, folds, arima
    )
    results: dict[str, list[HybridFoldResult]] = {"current_no_news": baseline_results}
    for candidate_id, (candidate, data) in candidates.items():
        results[candidate_id] = _candidate_results(candidate, data, folds, arima)
    elapsed = time.perf_counter() - started
    ranking = _ranking(results, baseline_results)
    predictions = _prediction_rows(results)
    eligible = ranking[ranking["eligible_for_future_shadow"]]
    winner = str((eligible if not eligible.empty else ranking).iloc[0]["candidate_id"])
    bootstrap = (
        None
        if winner == "current_no_news"
        else _block_bootstrap(results[winner], baseline_results)
    )

    validation_dates = pd.DatetimeIndex(base_data.dates[folds[0].validation_start : development_end])
    news_validation = news_snapshot.frame.set_index("date").loc[validation_dates]
    data_quality = {
        "validation_origins": len(news_validation),
        "origins_with_news": int((news_validation["news_no_news"] == 0.0).sum()),
        "no_news_rate": float(news_validation["news_no_news"].mean()),
        "mean_source_coverage": float(news_validation["news_source_coverage"].mean()),
        "mean_article_count_log1p": float(
            news_validation["news_article_count_log1p"].mean()
        ),
    }
    ranking.to_csv(args.output_dir / "ranking.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    fold_rows = pd.DataFrame(
        [
            {"candidate_id": candidate_id, **result.summary_row()}
            for candidate_id, candidate_results in results.items()
            for result in candidate_results
        ]
    )
    fold_rows.to_csv(args.output_dir / "folds.csv", index=False)
    _plot(args.output_dir, ranking, predictions, winner)
    payload: dict[str, object] = {
        "scope": "Diesel S10 news-signal challenger — frozen development folds only",
        "holdout_evaluated": False,
        "production_promotion_allowed": False,
        "development_end_index": development_end,
        "folds": [asdict(fold) for fold in folds],
        "original_folds": [asdict(fold) for fold in original_folds],
        "removed_nonweekly_pairs": removed_nonweekly_pairs,
        "news_dataset_sha256": news_snapshot.dataset_sha256,
        "parent_news_dataset_sha256": news_snapshot.parent_news_dataset_sha256,
        "source_catalog_sha256": news_snapshot.source_catalog_sha256,
        "data_quality": data_quality,
        "elapsed_seconds": elapsed,
        "candidate_count_including_current": len(results),
        "selected_for_future_shadow": winner if winner != "current_no_news" else None,
        "selection_gate": (
            "beats current and ARIMA in every development fold, with dictionary "
            "replacement rate <= 0.40"
        ),
        "bootstrap_vs_current": bootstrap,
        "ranking": ranking.to_dict(orient="records"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Backtest de notícias — Diesel S10",
        "",
        "O holdout final não foi reaberto. Este relatório não autoriza promoção.",
        "",
        f"- Dataset semanal v3: `{news_snapshot.dataset_sha256}`",
        f"- Folds: {len(folds)} x 52 semanas ({sum(f.validation_end-f.validation_start for f in folds)} previsões)",
        f"- Candidato selecionado para shadow futuro: `{payload['selected_for_future_shadow']}`",
        f"- Tempo: {elapsed:.2f} s",
        "",
        "## Ranking",
        "",
        _markdown_table(ranking),
        "",
        "## Qualidade do sinal nos folds",
        "",
        "```json",
        json.dumps(data_quality, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Decisão",
        "",
        "Somente um candidato que passe o gate em todos os folds pode seguir para shadow prospectivo. Mesmo assim, dados futuros ainda não observados são obrigatórios antes de qualquer uso operacional.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
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
    parser.add_argument("--news-signals", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "vs_epl_krls" / "s10_news",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
