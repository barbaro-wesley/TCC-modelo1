"""Causal text-pressure features from governed DieselNews snapshots.

The classifier in this module is deliberately *prequential*: before predicting
the pressure for an origin, it learns only labels whose target dates are no
later than that origin.  Market-movement labels are weak supervision and must
not be presented as analyst annotations or causal effects of a news article.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .selection import S10Supervised

PRESSURE_CLASSES = (-1, 0, 1)
NEWS_PRESSURE_FEATURES = (
    "news_pressure_down_probability",
    "news_pressure_neutral_probability",
    "news_pressure_up_probability",
    "news_pressure_score",
    "news_pressure_confidence",
    "news_pressure_entropy",
    "news_pressure_no_news",
    "news_pressure_article_count_log1p",
)

_POINTER_FIELDS = {
    "dataset_sha256",
    "pointer_sha256",
    "pointer_version",
    "relative_path",
    "schema_version",
}
_MANIFEST_FIELDS = {
    "config_sha256",
    "dataset_sha256",
    "dataset_type",
    "files",
    "manifest_version",
    "parent_provenance_dataset_sha256",
    "pipeline_version",
    "record_count",
    "retrieved_through",
    "schema_version",
    "source_catalog_sha256",
    "unavailable_count",
}
_RECORD_FIELDS = {
    "availability_basis",
    "canonical_url",
    "content_sha256",
    "first_available_at",
    "impact",
    "language",
    "license_name",
    "provenance_ids",
    "published_at",
    "published_date",
    "record_id",
    "response_sha256",
    "retrieved_at",
    "schema_version",
    "source_id",
    "storage_scope",
    "summary",
    "title",
    "updated_at",
}
_IMPACT_FIELDS = {
    "categories",
    "confidence",
    "direction",
    "evidence_terms",
    "horizon",
    "intensity",
    "model_version",
    "relevance_s10",
}
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsCorpusSnapshot:
    """Verified news-record.v2 frame with immutable lineage."""

    articles: pd.DataFrame
    dataset_sha256: str
    parent_provenance_dataset_sha256: str
    source_catalog_sha256: str
    retrieved_through: pd.Timestamp


@dataclass(frozen=True)
class WeeklyNewsDocuments:
    """One causal text document for each S10 forecast origin."""

    frame: pd.DataFrame
    news_dataset_sha256: str
    lookback_days: int
    min_relevance_s10: float


@dataclass(frozen=True)
class NewsPressureConfig:
    """Configuration for the deterministic online text classifier."""

    n_features: int = 2**14
    ngram_min: int = 1
    ngram_max: int = 2
    alpha: float = 1e-4
    min_training_samples: int = 52
    probability_shrinkage: float = 0.20
    neutral_threshold: float = 0.010
    average_weights: bool = False
    random_state: int = 20260823

    def __post_init__(self) -> None:
        if self.n_features < 256:
            raise ValueError("n_features must be at least 256")
        if not 1 <= self.ngram_min <= self.ngram_max <= 3:
            raise ValueError("ngram range must satisfy 1 <= min <= max <= 3")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if self.min_training_samples < 3:
            raise ValueError("min_training_samples must be at least 3")
        if not 0.0 <= self.probability_shrinkage <= 1.0:
            raise ValueError("probability_shrinkage must be in [0, 1]")
        if not math.isfinite(self.neutral_threshold) or self.neutral_threshold < 0:
            raise ValueError("neutral_threshold must be finite and non-negative")
        if not isinstance(self.average_weights, bool):
            raise TypeError("average_weights must be boolean")


@dataclass(frozen=True)
class NewsPressureFeatures:
    """Causal pressure probabilities aligned with an S10 supervised table."""

    frame: pd.DataFrame
    news_dataset_sha256: str
    config: NewsPressureConfig
    label_method: str = "future_s10_delta_weak_supervision"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _json_object(data: bytes, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _regular_file(path: Path, *, context: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} is not a regular file: {path}")
    return path.read_bytes()


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _unit_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return result


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _resolve_news_snapshot(path: Path) -> tuple[Path, dict[str, Any]]:
    if (path / "manifest.json").is_file():
        snapshot = path
    else:
        pointer_path = path if path.name == "latest.json" else path / "latest.json"
        pointer = _json_object(
            _regular_file(pointer_path, context="news-record latest pointer"),
            context="news-record latest pointer",
        )
        if set(pointer) != _POINTER_FIELDS:
            raise ValueError("news-record pointer fields differ from the v1 contract")
        if pointer.get("pointer_version") != "dataset-pointer.v1":
            raise ValueError("unsupported news-record pointer version")
        if pointer.get("schema_version") != "news-record.v2":
            raise ValueError("news-record pointer schema is not v2")
        dataset_hash = _sha256(pointer.get("dataset_sha256"), field="dataset_sha256")
        pointer_hash = _sha256(pointer.get("pointer_sha256"), field="pointer_sha256")
        core = {key: value for key, value in pointer.items() if key != "pointer_sha256"}
        if _digest(_canonical_json(core)) != pointer_hash:
            raise ValueError("news-record pointer checksum mismatch")
        relative_path = f"snapshots/{dataset_hash}"
        if pointer.get("relative_path") != relative_path:
            raise ValueError("news-record pointer path is not canonical")
        snapshot = pointer_path.parent / "snapshots" / dataset_hash
    manifest = _json_object(
        _regular_file(snapshot / "manifest.json", context="news-record manifest"),
        context="news-record manifest",
    )
    return snapshot, manifest


def _validate_impact(value: object, *, line_number: int) -> float:
    if not isinstance(value, dict) or set(value) != _IMPACT_FIELDS:
        raise ValueError(f"impact fields differ from the v2 contract at line {line_number}")
    if value.get("direction") not in {"down", "neutral", "uncertain", "up"}:
        raise ValueError(f"invalid impact direction at line {line_number}")
    for field in ("confidence", "intensity", "relevance_s10"):
        _unit_float(value.get(field), field=f"impact.{field}")
    for field in ("categories", "evidence_terms"):
        items = value.get(field)
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"impact.{field} must be a list of strings")
    for field in ("horizon", "model_version"):
        _text(value.get(field), field=f"impact.{field}")
    return float(value["relevance_s10"])


def _article_row(value: dict[str, Any], *, line_number: int) -> dict[str, object]:
    if set(value) != _RECORD_FIELDS:
        raise ValueError(f"news-record fields differ from v2 at line {line_number}")
    if value.get("schema_version") != "news-record.v2":
        raise ValueError(f"unexpected news-record schema at line {line_number}")
    record_id = _text(value.get("record_id"), field="record_id")
    if not record_id.startswith("news2:"):
        raise ValueError(f"invalid record_id at line {line_number}")
    first_available = pd.to_datetime(
        value.get("first_available_at"), utc=True, errors="raise"
    )
    if pd.isna(first_available):
        raise ValueError(f"invalid first_available_at at line {line_number}")
    impact = value.get("impact")
    relevance = _validate_impact(impact, line_number=line_number)
    assert isinstance(impact, dict)
    provenance = value.get("provenance_ids")
    if not isinstance(provenance, list) or not provenance or not all(
        isinstance(item, str) and item for item in provenance
    ):
        raise ValueError(f"invalid provenance_ids at line {line_number}")
    return {
        "record_id": record_id,
        "canonical_url": _text(value.get("canonical_url"), field="canonical_url"),
        "first_available_at": first_available,
        "published_date": _text(value.get("published_date"), field="published_date"),
        "title": _text(value.get("title"), field="title"),
        "summary": _text(value.get("summary"), field="summary", allow_empty=True),
        "source_id": _text(value.get("source_id"), field="source_id"),
        "license_name": _text(value.get("license_name"), field="license_name"),
        "content_sha256": _sha256(value.get("content_sha256"), field="content_sha256"),
        "relevance_s10": relevance,
        "machine_direction": str(impact["direction"]),
        "machine_confidence": float(impact["confidence"]),
        "machine_intensity": float(impact["intensity"]),
        "machine_categories": tuple(str(item) for item in impact["categories"]),
    }


def load_news_corpus(path: Path) -> NewsCorpusSnapshot:
    """Verify and load an immutable ``news-record.v2`` snapshot."""

    snapshot, manifest = _resolve_news_snapshot(path)
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("news-record manifest fields differ from the v2 contract")
    expected_versions = (
        manifest.get("manifest_version") == "dataset-manifest.v2"
        and manifest.get("dataset_type") == "news"
        and manifest.get("schema_version") == "news-record.v2"
        and manifest.get("pipeline_version") == "temporal-impact-collection.v1"
    )
    if not expected_versions:
        raise ValueError("unsupported news-record manifest version/type/schema/pipeline")
    dataset_hash = _sha256(manifest.get("dataset_sha256"), field="dataset_sha256")
    parent_hash = _sha256(
        manifest.get("parent_provenance_dataset_sha256"),
        field="parent_provenance_dataset_sha256",
    )
    source_hash = _sha256(
        manifest.get("source_catalog_sha256"), field="source_catalog_sha256"
    )
    _sha256(manifest.get("config_sha256"), field="config_sha256")
    core = {key: value for key, value in manifest.items() if key != "dataset_sha256"}
    if _digest(_canonical_json(core)) != dataset_hash or snapshot.name != dataset_hash:
        raise ValueError("news-record manifest content address mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"records.jsonl"}:
        raise ValueError("news-record manifest must declare only records.jsonl")
    member_hash = _sha256(files.get("records.jsonl"), field="files.records.jsonl")
    member = _regular_file(snapshot / "records.jsonl", context="news records")
    if _digest(member) != member_hash:
        raise ValueError("news records SHA-256 mismatch")
    if {item.name for item in snapshot.iterdir()} != {"manifest.json", "records.jsonl"}:
        raise ValueError("news-record snapshot contains undeclared files")

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(member.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank news record at line {line_number}")
        row = _article_row(
            _json_object(line, context=f"news record line {line_number}"),
            line_number=line_number,
        )
        record_id = str(row["record_id"])
        if record_id in seen_ids:
            raise ValueError(f"duplicate news record_id at line {line_number}")
        seen_ids.add(record_id)
        rows.append(row)
    expected_count = _nonnegative_integer(manifest.get("record_count"), field="record_count")
    if len(rows) != expected_count:
        raise ValueError("news-record manifest record_count mismatch")
    _nonnegative_integer(manifest.get("unavailable_count"), field="unavailable_count")
    retrieved_through = pd.to_datetime(
        manifest.get("retrieved_through"), utc=True, errors="raise"
    )
    articles = pd.DataFrame(rows).sort_values(
        ["first_available_at", "record_id"], ignore_index=True
    )
    if articles.empty:
        raise ValueError("news-record snapshot is empty")
    return NewsCorpusSnapshot(
        articles=articles,
        dataset_sha256=dataset_hash,
        parent_provenance_dataset_sha256=parent_hash,
        source_catalog_sha256=source_hash,
        retrieved_through=retrieved_through,
    )


def _normalized_text(value: str, *, limit: int) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    return _SPACE.sub(" ", normalized).strip()[:limit]


def build_weekly_news_documents(
    snapshot: NewsCorpusSnapshot,
    origin_dates: Iterable[object],
    *,
    lookback_days: int = 28,
    min_relevance_s10: float = 0.0,
    max_article_characters: int = 3_000,
) -> WeeklyNewsDocuments:
    """Aggregate only articles available by each midnight UTC origin."""

    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    if not 0.0 <= min_relevance_s10 <= 1.0:
        raise ValueError("min_relevance_s10 must be in [0, 1]")
    if max_article_characters < 100:
        raise ValueError("max_article_characters must be at least 100")
    origins = pd.DatetimeIndex(pd.to_datetime(list(origin_dates), errors="raise"))
    if origins.tz is None:
        origins = origins.tz_localize("UTC")
    else:
        origins = origins.tz_convert("UTC")
    if origins.has_duplicates or not origins.is_monotonic_increasing:
        raise ValueError("origin_dates must be strictly increasing and unique")
    if not (origins == origins.normalize()).all():
        raise ValueError("origin_dates must be normalized to midnight")

    articles = snapshot.articles
    eligible = articles[articles["relevance_s10"] >= min_relevance_s10].copy()
    rows: list[dict[str, object]] = []
    window = pd.Timedelta(days=lookback_days)
    for origin in origins:
        selected = eligible[
            (eligible["first_available_at"] > origin - window)
            & (eligible["first_available_at"] <= origin)
        ].drop_duplicates("content_sha256", keep="first")
        parts: list[str] = []
        for article in selected.itertuples(index=False):
            title = _normalized_text(str(article.title), limit=max_article_characters)
            summary = _normalized_text(str(article.summary), limit=max_article_characters)
            source = re.sub(r"[^a-z0-9_]+", "_", str(article.source_id).lower())
            text = f"source_{source} {title}"
            if summary:
                text += f" {summary}"
            parts.append(text[:max_article_characters])
        rows.append(
            {
                "date": origin.tz_localize(None),
                "document": "\n".join(parts) if parts else "__no_news__",
                "article_count": len(parts),
                "no_news": bool(not parts),
            }
        )
    return WeeklyNewsDocuments(
        frame=pd.DataFrame(rows),
        news_dataset_sha256=snapshot.dataset_sha256,
        lookback_days=int(lookback_days),
        min_relevance_s10=float(min_relevance_s10),
    )


def market_pressure_labels(
    data: S10Supervised, *, neutral_threshold: float = 0.005
) -> NDArray[np.int8]:
    """Create transparent weak labels from the later realized S10 movement."""

    if not math.isfinite(neutral_threshold) or neutral_threshold < 0:
        raise ValueError("neutral_threshold must be finite and non-negative")
    delta = np.asarray(data.target_price - data.origin_price, dtype=float)
    if not np.all(np.isfinite(delta)):
        raise ValueError("S10 target deltas must be finite")
    labels = np.zeros(data.n_samples, dtype=np.int8)
    labels[delta < -neutral_threshold] = -1
    labels[delta > neutral_threshold] = 1
    return labels


class OnlineNewsPressureClassifier:
    """Deterministic hashed-text logistic classifier with ``learn_one`` API."""

    def __init__(self, config: NewsPressureConfig | None = None) -> None:
        self.config = config or NewsPressureConfig()
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            from sklearn.linear_model import SGDClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "scikit-learn is required for news pressure classification; "
                "install the 'production' or 'experiments' extra"
            ) from exc
        self._vectorizer = HashingVectorizer(
            n_features=self.config.n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
        )
        self._classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=self.config.alpha,
            fit_intercept=True,
            shuffle=False,
            average=self.config.average_weights,
            random_state=self.config.random_state,
        )
        self._classes = np.asarray(PRESSURE_CLASSES, dtype=np.int8)
        self._class_counts = np.zeros(3, dtype=np.int64)
        self._initialized = False
        self.n_samples_seen_ = 0

    def _prior(self) -> NDArray[np.float64]:
        smoothed = self._class_counts.astype(float) + 1.0
        return smoothed / float(smoothed.sum())

    def predict_proba_one(self, document: str) -> NDArray[np.float64]:
        """Return probabilities ordered as down, neutral and up."""

        if not isinstance(document, str) or not document.strip():
            raise ValueError("document must be a non-empty string")
        prior = self._prior()
        if not self._initialized or self.n_samples_seen_ < self.config.min_training_samples:
            return prior
        matrix = self._vectorizer.transform([document])
        raw = np.asarray(self._classifier.predict_proba(matrix)[0], dtype=float)
        order = {int(label): index for index, label in enumerate(self._classifier.classes_)}
        probabilities = np.asarray([raw[order[label]] for label in PRESSURE_CLASSES])
        shrinkage = self.config.probability_shrinkage
        probabilities = (1.0 - shrinkage) * probabilities + shrinkage * prior
        probabilities = np.clip(probabilities, 1e-12, 1.0)
        return probabilities / float(probabilities.sum())

    def learn_one(self, document: str, label: int) -> OnlineNewsPressureClassifier:
        """Update after the corresponding target date has become observable."""

        if not isinstance(document, str) or not document.strip():
            raise ValueError("document must be a non-empty string")
        if isinstance(label, bool) or int(label) not in PRESSURE_CLASSES:
            raise ValueError("label must be one of -1, 0, 1")
        class_index = PRESSURE_CLASSES.index(int(label))
        matrix = self._vectorizer.transform([document])
        if self._initialized:
            self._classifier.partial_fit(matrix, np.asarray([label], dtype=np.int8))
        else:
            self._classifier.partial_fit(
                matrix,
                np.asarray([label], dtype=np.int8),
                classes=self._classes,
            )
            self._initialized = True
        self._class_counts[class_index] += 1
        self.n_samples_seen_ += 1
        return self

    def reset(self) -> None:
        """Restore the exact initial state while retaining the configuration."""

        replacement = OnlineNewsPressureClassifier(self.config)
        self._vectorizer = replacement._vectorizer
        self._classifier = replacement._classifier
        self._classes = replacement._classes
        self._class_counts = replacement._class_counts
        self._initialized = replacement._initialized
        self.n_samples_seen_ = replacement.n_samples_seen_


def generate_prequential_pressure_features(
    data: S10Supervised,
    documents: WeeklyNewsDocuments,
    *,
    config: NewsPressureConfig | None = None,
) -> NewsPressureFeatures:
    """Generate one leak-safe pressure row per supervised forecast origin."""

    classifier_config = config or NewsPressureConfig()
    frame = documents.frame
    origins = pd.DatetimeIndex(data.dates).normalize()
    document_dates = pd.DatetimeIndex(frame["date"]).normalize()
    if len(frame) != data.n_samples or not origins.equals(document_dates):
        raise ValueError("weekly documents must align exactly with supervised origins")
    labels = market_pressure_labels(
        data, neutral_threshold=classifier_config.neutral_threshold
    )
    classifier = OnlineNewsPressureClassifier(classifier_config)
    learned_until = -1
    rows: list[dict[str, object]] = []
    for index in range(data.n_samples):
        origin = data.dates[index]
        while (
            learned_until + 1 < data.n_samples
            and data.target_dates[learned_until + 1] <= origin
        ):
            learned_until += 1
            classifier.learn_one(
                str(frame.iloc[learned_until]["document"]),
                int(labels[learned_until]),
            )
        probabilities = classifier.predict_proba_one(str(frame.iloc[index]["document"]))
        entropy = -float(np.sum(probabilities * np.log(probabilities))) / math.log(3.0)
        rows.append(
            {
                "date": pd.Timestamp(origin).normalize(),
                "news_pressure_down_probability": float(probabilities[0]),
                "news_pressure_neutral_probability": float(probabilities[1]),
                "news_pressure_up_probability": float(probabilities[2]),
                "news_pressure_score": float(probabilities[2] - probabilities[0]),
                "news_pressure_confidence": float(np.max(probabilities)),
                "news_pressure_entropy": entropy,
                "news_pressure_no_news": float(bool(frame.iloc[index]["no_news"])),
                "news_pressure_article_count_log1p": math.log1p(
                    int(frame.iloc[index]["article_count"])
                ),
                "pressure_true_class": int(labels[index]),
                "pressure_predicted_class": int(PRESSURE_CLASSES[int(np.argmax(probabilities))]),
                "pressure_training_samples": int(classifier.n_samples_seen_),
            }
        )
    output = pd.DataFrame(rows)
    feature_values = output[list(NEWS_PRESSURE_FEATURES)].to_numpy(dtype=float)
    if not np.all(np.isfinite(feature_values)):
        raise RuntimeError("generated news pressure features contain NaN or infinity")
    return NewsPressureFeatures(
        frame=output,
        news_dataset_sha256=documents.news_dataset_sha256,
        config=classifier_config,
    )


def augment_supervised_with_pressure(
    base: S10Supervised,
    pressure: NewsPressureFeatures,
    *,
    feature_names: tuple[str, ...] = NEWS_PRESSURE_FEATURES,
) -> S10Supervised:
    """Append prequential pressure features without changing target alignment."""

    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique")
    missing = [name for name in feature_names if name not in pressure.frame]
    if missing:
        raise ValueError(f"pressure frame is missing features: {missing}")
    dates = pd.DatetimeIndex(pressure.frame["date"]).normalize()
    origins = pd.DatetimeIndex(base.dates).normalize()
    if len(dates) != base.n_samples or not origins.equals(dates):
        raise ValueError("pressure features must align exactly with supervised origins")
    values = pressure.frame[list(feature_names)].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("pressure feature matrix contains NaN or infinity")
    return S10Supervised(
        x=np.column_stack((base.x, values)),
        target_price=base.target_price.copy(),
        origin_price=base.origin_price.copy(),
        dates=base.dates.copy(),
        target_dates=base.target_dates.copy(),
        feature_names=(*base.feature_names, *feature_names),
        horizon=base.horizon,
    )
