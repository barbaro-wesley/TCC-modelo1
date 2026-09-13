"""Leak-safe preparation and evaluation of weekly ANP fuel-price CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import time
import unicodedata
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd

from .metrics import regression_report
from .model import VSEPLKRLS
from .utils import MinMaxScaler


def _normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


_DATE_ALIASES = {
    "data",
    "date",
    "data_coleta",
    "data_inicial",
    "data_final",
    "semana",
    "week",
}
_PRODUCT_ALIASES = {"produto", "product", "combustivel", "descricao_produto"}
_PRICE_ALIASES = {
    "preco",
    "price",
    "preco_medio",
    "preco_medio_revenda",
    "valor_venda",
    "valor_revenda",
}


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    aliases: set[str],
    kind: str,
) -> str:
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(f"{kind} column {explicit!r} was not found")
        return explicit
    normalized = {_normalized_name(column): str(column) for column in frame.columns}
    if kind == "date" and "data_inicial" in normalized:
        return normalized["data_inicial"]
    matches = [normalized[name] for name in aliases if name in normalized]
    if not matches and kind == "price":
        matches = [
            original
            for name, original in normalized.items()
            if "revenda" in name and name.startswith(("preco", "pre_o"))
        ]
    if len(matches) != 1:
        raise ValueError(
            f"could not uniquely identify the {kind} column; pass it explicitly. "
            f"Available columns: {list(frame.columns)}"
        )
    return matches[0]


def _parse_prices(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.replace(r"R\$\s*", "", regex=True)
    comma = text.str.contains(",", regex=False)
    text.loc[comma] = (
        text.loc[comma].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def canonical_product(
    value: object,
    *,
    generic_diesel_as_s500: bool = False,
) -> str | None:
    """Map common ANP diesel descriptions to ``S10`` or ``S500``."""

    text = _normalized_name(value).replace("_", "")
    if re.search(r"s0*500", text):
        return "S500"
    if re.search(r"s0*10", text):
        return "S10"
    if generic_diesel_as_s500 and text in {"oleodiesel", "diesel"}:
        return "S500"
    return None


def _read_anp_file(source: Path, options: dict[str, object]) -> pd.DataFrame:
    if source.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(source, header=None, **options)
        header_row: int | None = None
        for index, row in raw.iterrows():
            names = {_normalized_name(value) for value in row.dropna().tolist()}
            if "produto" in names and ("data_inicial" in names or "data" in names):
                header_row = int(index)
                break
        if header_row is None:
            raise ValueError("could not find the ANP spreadsheet header")
        frame = raw.iloc[header_row + 1 :].copy()
        frame.columns = [str(value).strip() for value in raw.iloc[header_row].tolist()]
        return frame.dropna(how="all")
    csv_defaults: dict[str, object] = {"sep": None, "engine": "python"}
    csv_defaults.update(options)
    return pd.read_csv(source, **csv_defaults)


def load_anp_fuel_csv(
    path: str | Path,
    *,
    date_column: str | None = None,
    product_column: str | None = None,
    price_column: str | None = None,
    products: Iterable[str] = ("S10", "S500"),
    weekly: Literal["preserve", "mean", "median"] = "mean",
    week_ending: str = "SUN",
    csv_options: dict[str, object] | None = None,
    generic_diesel_as_s500: bool = False,
) -> pd.DataFrame:
    """Load, validate, filter and optionally aggregate a local ANP CSV.

    No network download or fabricated fallback is performed.  The returned
    columns are ``date``, ``product`` and ``price``.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"ANP CSV not found: {source}")
    frame = _read_anp_file(source, dict(csv_options or {}))
    if frame.empty:
        raise ValueError("the ANP CSV is empty")
    date_name = _resolve_column(frame, date_column, _DATE_ALIASES, "date")
    product_name = _resolve_column(frame, product_column, _PRODUCT_ALIASES, "product")
    price_name = _resolve_column(frame, price_column, _PRICE_ALIASES, "price")

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_name], errors="coerce", dayfirst=True),
            "product": frame[product_name].map(
                lambda value: canonical_product(
                    value,
                    generic_diesel_as_s500=generic_diesel_as_s500,
                )
            ),
            "price": _parse_prices(frame[price_name]),
        }
    )
    requested = {str(product).upper() for product in products}
    invalid_requested = requested - {"S10", "S500"}
    if invalid_requested:
        raise ValueError(f"unsupported products: {sorted(invalid_requested)}")
    result = result[result["product"].isin(requested)].dropna().copy()
    if result.empty:
        raise ValueError("no valid S10/S500 observations remained after validation")
    if not np.all(np.isfinite(result["price"])) or (result["price"] <= 0).any():
        raise ValueError("fuel prices must be finite and positive")

    result = result.sort_values(["product", "date"]).reset_index(drop=True)
    if weekly != "preserve":
        reducer = "mean" if weekly == "mean" else "median"
        result["date"] = result["date"].dt.to_period(f"W-{week_ending}").dt.end_time.dt.normalize()
        result = (
            result.groupby(["product", "date"], as_index=False)["price"]
            .agg(reducer)
            .sort_values(["product", "date"])
            .reset_index(drop=True)
        )
    elif result.duplicated(["product", "date"]).any():
        raise ValueError("duplicate product/date rows require weekly='mean' or 'median'")
    return result


def make_lagged_dataset(
    frame: pd.DataFrame,
    *,
    lags: Iterable[int] = (0, 1, 2, 3, 4, 8, 12),
    horizon: int = 2,
) -> pd.DataFrame:
    """Build origin-time lag features and a future target without leakage."""

    required = {"date", "product", "price"}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame must contain columns {sorted(required)}")
    lag_values = sorted({int(lag) for lag in lags})
    if not lag_values or lag_values[0] < 0:
        raise ValueError("lags must contain non-negative integers")
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    rows: list[pd.DataFrame] = []
    for product, group in frame.groupby("product", sort=True):
        group = group.sort_values("date").reset_index(drop=True).copy()
        output = pd.DataFrame({"date": group["date"], "product": product, "price": group["price"]})
        for lag in lag_values:
            output[f"lag_{lag}"] = group["price"].shift(lag)
        output["target"] = group["price"].shift(-horizon)
        output["target_date"] = group["date"].shift(-horizon)
        rows.append(output.dropna())
    if not rows:
        raise ValueError("no product groups were available")
    result = pd.concat(rows, ignore_index=True)
    if result.empty:
        raise ValueError("not enough rows for the requested lags and horizon")
    return result.sort_values(["product", "date"]).reset_index(drop=True)


@dataclass
class FuelEvaluation:
    product: str
    horizon: int
    split_date: str
    metrics: dict[str, float]
    naive_metrics: dict[str, float]
    model_summary: dict[str, object]
    elapsed_seconds: float
    learning_seconds: float
    prediction_seconds: float
    n_learning_updates: int
    n_test_predictions: int
    predictions: pd.DataFrame

    def summary_row(self) -> dict[str, object]:
        return {
            "product": self.product,
            "horizon_weeks": self.horizon,
            "split_date": self.split_date,
            **self.metrics,
            "naive_rmse": self.naive_metrics["rmse"],
            "final_rules": self.model_summary["n_rules"],
            "max_rules": self.model_summary["max_rules_observed"],
            "rule_creations": self.model_summary["rule_creations"],
            "rule_merges": self.model_summary["rule_merges"],
            "mean_dictionary_size": self.model_summary["mean_dictionary_size"],
            "max_dictionary_size": self.model_summary["max_dictionary_size"],
            "elapsed_seconds": self.elapsed_seconds,
            "learning_seconds": self.learning_seconds,
            "prediction_seconds": self.prediction_seconds,
            "n_learning_updates": self.n_learning_updates,
            "n_test_predictions": self.n_test_predictions,
        }


def evaluate_fuel_product(
    lagged: pd.DataFrame,
    *,
    product: str,
    horizon: int,
    train_fraction: float = 0.8,
    model_factory: Callable[[], VSEPLKRLS] | None = None,
) -> FuelEvaluation:
    """Run delayed prequential evaluation on one product.

    A target from origin ``k`` is learned only at origin ``k+horizon``. This is
    stricter than immediately feeding each offline test label back to the model.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    canonical = product.upper()
    data = lagged[lagged["product"] == canonical].sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError(f"no rows for product {canonical}")
    feature_names = [column for column in data.columns if column.startswith("lag_")]
    if not feature_names:
        raise ValueError("lagged data contains no lag feature columns")
    split = int(np.floor(len(data) * train_fraction))
    if split <= horizon or split >= len(data):
        raise ValueError("split leaves too few rows for delayed evaluation")

    x_raw = data[feature_names].to_numpy(float)
    y_raw = data["target"].to_numpy(float)
    x_scaler = MinMaxScaler().fit(x_raw[:split])
    known_target_end = split - horizon
    y_scaler = MinMaxScaler().fit(y_raw[:known_target_end])
    x_scaled = np.clip(x_scaler.transform(x_raw), 0.0, 1.0)
    y_scaled = y_scaler.transform(y_raw)

    model = model_factory() if model_factory is not None else VSEPLKRLS()
    started = time.perf_counter()
    learning_started = time.perf_counter()
    learned_until = -1
    for index in range(known_target_end):
        model.learn_one(x_scaled[index], float(y_scaled[index]))
        learned_until = index
    learning_seconds = time.perf_counter() - learning_started

    test_predictions: list[float] = []
    rule_counts: list[int] = []
    betas: list[float] = []
    prediction_seconds = 0.0
    for index in range(split, len(data)):
        newly_available = index - horizon
        if newly_available > learned_until:
            update_started = time.perf_counter()
            model.learn_one(x_scaled[newly_available], float(y_scaled[newly_available]))
            learning_seconds += time.perf_counter() - update_started
            learned_until = newly_available
        prediction_started = time.perf_counter()
        test_predictions.append(model.predict_one(x_scaled[index]))
        prediction_seconds += time.perf_counter() - prediction_started
        rule_counts.append(model.n_rules)
        betas.append(model.beta_)
    elapsed = time.perf_counter() - started

    prediction = y_scaler.inverse_transform(np.asarray(test_predictions))
    truth = y_raw[split:]
    if "lag_0" in data:
        naive = data.loc[split:, "lag_0"].to_numpy(float)
    else:
        smallest_lag = min(feature_names, key=lambda value: int(value.split("_")[1]))
        naive = data.loc[split:, smallest_lag].to_numpy(float)
    prediction_frame = pd.DataFrame(
        {
            "origin_date": data.loc[split:, "date"].to_numpy(),
            "target_date": data.loc[split:, "target_date"].to_numpy(),
            "actual": truth,
            "prediction": prediction,
            "naive": naive,
            "n_rules": rule_counts,
            "beta": betas,
        }
    )
    return FuelEvaluation(
        product=canonical,
        horizon=int(horizon),
        split_date=str(pd.Timestamp(data.loc[split, "date"]).date()),
        metrics=regression_report(truth, prediction),
        naive_metrics=regression_report(truth, naive),
        model_summary=model.summary(),
        elapsed_seconds=float(elapsed),
        learning_seconds=float(learning_seconds),
        prediction_seconds=float(prediction_seconds),
        n_learning_updates=model.n_seen_,
        n_test_predictions=len(test_predictions),
        predictions=prediction_frame,
    )
