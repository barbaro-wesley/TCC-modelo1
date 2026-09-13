"""Research-only simulation of two independent S10 news annotators.

These deterministic policies exercise the annotation workflow and provide weak
labels for experimentation.  Their outputs are not human ground truth and are
never sufficient for production training or model promotion.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .news_annotation import ANNOTATION_COLUMNS

PersonaName = Literal[
    "supply_cost",
    "procurement_risk",
    "procurement_risk_calibrated",
]

_DIRECT = (
    "diesel s10",
    "diesel",
    "oleo diesel",
    "biodiesel",
)
_FUEL = ("combustivel", "derivados", "gasoleo", "abastecimento")
_SECTOR = (
    "petroleo",
    "brent",
    "refino",
    "refinaria",
    "petrobras",
    "barril",
    "opep",
    "importacao",
    "exportacao",
    "producao",
    "estoque",
)
_COST = (
    "dolar",
    "cambio",
    "tribut",
    "imposto",
    "icms",
    "pis/cofins",
    "frete",
    "logistica",
)
_UP = (
    "aumento de preco",
    "aumento do preco",
    "alta de preco",
    "alta do preco",
    "reajuste",
    "eleva imposto",
    "aumento de imposto",
    "aumenta imposto",
    "corte de producao",
    "reduz producao",
    "reducao da producao",
    "reduz oferta",
    "reducao da oferta",
    "queda da oferta",
    "paralisacao",
    "greve",
    "interrupcao",
    "escassez",
    "embargo",
    "sancoes",
    "conflito",
    "desvalorizacao do real",
    "alta do petroleo",
    "petroleo sobe",
    "dolar sobe",
)
_DOWN = (
    "reducao de preco",
    "reducao do preco",
    "queda de preco",
    "queda do preco",
    "reduz imposto",
    "reducao de imposto",
    "corte de imposto",
    "desoneracao",
    "isencao",
    "subsidio",
    "aumenta producao",
    "aumento da producao",
    "amplia producao",
    "aumenta oferta",
    "ampliacao da oferta",
    "amplia oferta",
    "alta da oferta",
    "valorizacao do real",
    "queda do petroleo",
    "petroleo cai",
    "dolar cai",
)
_NEUTRAL_POLICY = (
    "consulta publica",
    "audiencia publica",
    "estudo",
    "seminario",
    "debate",
    "grupo de trabalho",
    "relatorio",
    "monitoramento",
    "agenda",
)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SimulatedLabel:
    """One transparent simulated annotation decision."""

    relevance: int
    direction: str
    intensity: int
    horizon: str
    evidence: str
    rationale: str


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.lower())
    ascii_text = "".join(character for character in decomposed if not unicodedata.combining(character))
    return _SPACE.sub(" ", ascii_text).strip()


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def _relevance(text: str, *, persona: PersonaName) -> tuple[int, list[str]]:
    direct = _matches(text, _DIRECT)
    fuel = _matches(text, _FUEL)
    sector = _matches(text, _SECTOR)
    cost = _matches(text, _COST)
    evidence = list(dict.fromkeys([*direct, *fuel, *sector, *cost]))
    if persona == "supply_cost":
        if direct:
            return 3, evidence
        if fuel or (sector and cost):
            return 2, evidence
        if sector or cost:
            return 1, evidence
        return 0, evidence
    if persona == "procurement_risk_calibrated":
        if direct:
            return 3, evidence
        if fuel or (sector and cost):
            return 2, evidence
        if sector or cost:
            # Calibration accepts indirect market links, but keeps purely
            # informational events outside the actionable procurement scope.
            return (0 if _matches(text, _NEUTRAL_POLICY) else 1), evidence
        return 0, evidence
    if "diesel s10" in direct or "oleo diesel" in direct or "diesel" in direct:
        return 3, evidence
    if "biodiesel" in direct or fuel or (sector and cost):
        return 2, evidence
    if len(sector) + len(cost) >= 2:
        return 1, evidence
    return 0, evidence


def _direction(
    text: str,
    *,
    persona: PersonaName,
    relevance: int,
) -> tuple[str, list[str]]:
    up = _matches(text, _UP)
    down = _matches(text, _DOWN)
    evidence = list(dict.fromkeys([*up, *down]))
    if relevance == 0:
        return "neutral", evidence
    if up and down:
        return "uncertain", evidence
    if persona == "procurement_risk":
        explicit_context = bool(
            _matches(text, ("preco", "imposto", "producao", "oferta", "greve", "reajuste"))
        )
        if not explicit_context:
            if _matches(text, _NEUTRAL_POLICY):
                return "neutral", evidence
            return "uncertain", evidence
    if up:
        return "up", evidence
    if down:
        return "down", evidence
    if _matches(text, _NEUTRAL_POLICY):
        return "neutral", evidence
    return "uncertain", evidence


def _horizon(text: str, *, relevance: int, direction: str) -> str:
    if relevance == 0 or direction == "neutral":
        return "unknown"
    if _matches(text, ("reajuste", "preco", "imposto", "greve", "paralisacao")):
        return "1w"
    if _matches(text, ("oferta", "producao", "estoque", "importacao", "refinaria", "refino")):
        return "2w"
    if _matches(text, ("biodiesel", "mistura", "resolucao", "regulament", "lei")):
        return "4w"
    if _matches(text, ("investimento", "transicao", "plano", "2030", "2050")):
        return "long"
    return "unknown"


def simulate_label(text: str, *, persona: PersonaName) -> SimulatedLabel:
    """Apply one documented persona without consulting realized price data."""

    if persona not in {
        "supply_cost",
        "procurement_risk",
        "procurement_risk_calibrated",
    }:
        raise ValueError("unsupported simulation persona")
    normalized = _normalize(text)
    relevance, relevance_terms = _relevance(normalized, persona=persona)
    direction, direction_terms = _direction(
        normalized, persona=persona, relevance=relevance
    )
    if relevance == 0 or direction == "neutral":
        intensity = 0
    elif direction == "uncertain":
        intensity = 1
    else:
        intensity = min(
            3,
            relevance
            if persona == "supply_cost"
            else max(1, relevance - 1),
        )
    horizon = _horizon(normalized, relevance=relevance, direction=direction)
    terms = list(dict.fromkeys([*relevance_terms, *direction_terms]))[:6]
    evidence = "Termos detectados: " + ", ".join(terms) if terms else "Sem termo setorial suficiente."
    if relevance == 0:
        rationale = "A política simulada não encontrou mecanismo plausível ligado ao Diesel S10."
    elif direction == "uncertain":
        rationale = "Há relação setorial, mas o texto não define pressão direcional suficiente."
    elif direction == "neutral":
        rationale = "O tema é relacionado, porém informativo ou processual, sem efeito material explícito."
    else:
        verb = "elevação" if direction == "up" else "redução"
        rationale = f"Os termos indicam mecanismo plausível de {verb} de custo ou preço do S10."
    return SimulatedLabel(relevance, direction, intensity, horizon, evidence, rationale)


def simulate_annotation_slot(
    frame: pd.DataFrame,
    *,
    persona: PersonaName,
    annotated_at_utc: str = "2026-08-23T12:00:00Z",
) -> pd.DataFrame:
    """Fill one untouched slot file with a named simulated persona."""

    if tuple(frame.columns) != ANNOTATION_COLUMNS:
        raise ValueError("annotation columns differ from the v1 contract")
    slots = pd.to_numeric(frame["annotation_slot"], errors="coerce")
    if slots.isna().any() or slots.nunique() != 1:
        raise ValueError("simulation input must contain exactly one annotation slot")
    editable = ANNOTATION_COLUMNS[ANNOTATION_COLUMNS.index("annotator_id") :]
    if any(frame[column].astype("string").str.strip().ne("").any() for column in editable):
        raise ValueError("simulation only accepts an untouched blank annotation file")
    timestamp = pd.to_datetime(annotated_at_utc, utc=True, errors="raise")
    output = frame.copy()
    labels = [
        simulate_label(f"{row.title} {row.summary}", persona=persona)
        for row in output.itertuples(index=False)
    ]
    output["annotator_id"] = f"simulated_{persona}_v1"
    output["annotated_at_utc"] = timestamp.isoformat().replace("+00:00", "Z")
    output["relevance_label"] = [str(label.relevance) for label in labels]
    output["direction_label"] = [label.direction for label in labels]
    output["intensity_label"] = [str(label.intensity) for label in labels]
    output["horizon_label"] = [label.horizon for label in labels]
    output["evidence_text"] = [label.evidence for label in labels]
    output["rationale"] = [label.rationale for label in labels]
    return output
