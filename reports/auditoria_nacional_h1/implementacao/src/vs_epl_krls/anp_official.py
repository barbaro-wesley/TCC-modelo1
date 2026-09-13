"""Strict extraction of the official ANP national weekly S10 observation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


def _normalized(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^A-Z0-9$/]+", " ", text.upper()).strip()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ANPS10Observation:
    week_start: str
    week_end: str
    price_brl_per_liter: float
    stations_surveyed: int
    geography: str
    product: str
    unit: str
    source_path: str
    source_sha256: str
    source_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_national_s10_observation(
    workbook: str | Path,
    *,
    source_url: str | None = None,
) -> ANPS10Observation:
    """Extract exactly one national S10 row from an ANP weekly workbook.

    The official files have a nine-row report preamble and, historically, some
    mojibake in labels.  Identification therefore combines semantic cell
    contents with fixed relationships between the unit and resale-price fields.
    Every economically important field is validated before it is returned.
    """

    source = Path(workbook)
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        table = pd.read_excel(source, sheet_name="BRASIL", header=9)
    except (ValueError, ImportError) as exc:
        raise ValueError("ANP workbook must contain a readable BRASIL sheet") from exc
    if table.shape[1] < 7 or table.empty:
        raise ValueError("ANP BRASIL sheet has an unexpected schema")

    normalized_columns = [_normalized(column) for column in table.columns]
    product_candidates = [
        index
        for index, column in enumerate(table.columns)
        if table[column].astype(str).map(_normalized).str.contains("DIESEL").any()
    ]
    unit_candidates = [
        index
        for index, column in enumerate(table.columns)
        if table[column].astype(str).map(_normalized).str.replace(" ", "", regex=False).eq("R$/L").any()
    ]
    if len(product_candidates) != 1 or len(unit_candidates) != 1:
        raise ValueError("could not identify unique ANP product and unit columns")
    product_index = product_candidates[0]
    unit_index = unit_candidates[0]
    price_index = unit_index + 1
    stations_index = unit_index - 1
    if price_index >= table.shape[1] or stations_index < 0:
        raise ValueError("ANP price/station columns are not in the expected relationship")
    if "REVENDA" not in normalized_columns[price_index]:
        raise ValueError("ANP resale-price column was not found after unit column")

    product_column = table.columns[product_index]
    unit_column = table.columns[unit_index]
    normalized_products = table[product_column].astype(str).map(_normalized)
    normalized_units = (
        table[unit_column].astype(str).map(_normalized).str.replace(" ", "", regex=False)
    )
    mask = normalized_products.str.contains("DIESEL") & normalized_products.str.contains("S10")
    mask &= normalized_units.eq("R$/L")
    rows = table.loc[mask]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one national Diesel S10 row, found {len(rows)}")
    row = rows.iloc[0]

    week_start = pd.Timestamp(row.iloc[0])
    week_end = pd.Timestamp(row.iloc[1])
    if pd.isna(week_start) or pd.isna(week_end) or week_end - week_start != pd.Timedelta(days=6):
        raise ValueError("ANP weekly date range must span exactly seven calendar days")
    if week_start.dayofweek != 6:
        raise ValueError("ANP national week must start on Sunday")
    geography = _normalized(row.iloc[2])
    if geography != "BRASIL":
        raise ValueError("ANP observation is not national")
    price = float(pd.to_numeric(row.iloc[price_index], errors="coerce"))
    stations = int(pd.to_numeric(row.iloc[stations_index], errors="coerce"))
    if not np.isfinite(price) or not 1.0 <= price <= 30.0:
        raise ValueError("ANP S10 price is outside the accepted R$/L safety range")
    if stations < 1:
        raise ValueError("ANP station count must be positive")
    return ANPS10Observation(
        week_start=str(week_start.date()),
        week_end=str(week_end.date()),
        price_brl_per_liter=price,
        stations_surveyed=stations,
        geography="BRASIL",
        product="OLEO DIESEL S10",
        unit="R$/L",
        source_path=str(source.resolve()),
        source_sha256=sha256_file(source),
        source_url=source_url,
    )

