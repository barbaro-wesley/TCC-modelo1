from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import anp


def test_normalization_header_and_column_lookup():
    assert anp._norm(" Óleo Díesel ç ") == "OLEO DIESEL C"
    raw = pd.DataFrame([["relatorio", None], ["Mês", "Produto"], ["x", "y"]])
    assert anp._find_header_row(raw) == 1
    frame = pd.DataFrame(columns=["Preço Médio Revenda", "PRODUTO"])
    assert anp._col(frame, "PRECO MEDIO REVENDA") == "Preço Médio Revenda"
    with pytest.raises(KeyError, match="Coluna nao encontrada"):
        anp._col(frame, "inexistente")


def test_header_lookup_rejects_unknown_sheet():
    with pytest.raises(ValueError, match="cabecalho"):
        anp._find_header_row(pd.DataFrame([["foo"], ["bar"]]))


def test_read_anp_sheet_finds_embedded_header(monkeypatch):
    raw = pd.DataFrame(
        [
            ["Relatorio ANP", None, None],
            ["MES", "PRODUTO", "PREÇO MÉDIO REVENDA"],
            ["01/01/2020", "OLEO DIESEL S10", 4.1],
            [None, None, None],
        ]
    )
    monkeypatch.setattr(anp.pd, "read_excel", lambda *args, **kwargs: raw)
    result = anp.read_anp_sheet(Path("fake.xlsx"))
    assert list(result.columns) == ["MES", "PRODUTO", "PREÇO MÉDIO REVENDA"]
    assert len(result) == 1


def test_filter_s10_accepts_aliases_and_rejects_empty():
    frame = pd.DataFrame(
        {
            "PRODUTO": ["ÓLEO DIESEL S-10", "GASOLINA"],
            "valor": [1, 2],
        }
    )
    assert anp.filter_s10(frame)["valor"].tolist() == [1]
    with pytest.raises(ValueError, match="Nenhuma linha"):
        anp.filter_s10(frame.iloc[1:])


def test_parse_monthly_coerces_types_sorts_and_deduplicates(monkeypatch):
    source = pd.DataFrame(
        {
            "PRODUTO": ["DIESEL S10"] * 4,
            "MÊS": ["01/02/2020", "01/01/2020", "01/01/2020", "invalida"],
            "PRECO MEDIO REVENDA": [4.2, "4.0", 9.0, 3.0],
            "PRECO MEDIO DISTRIBUICAO": [3.8, 3.6, 8.0, 2.0],
        }
    )
    monkeypatch.setattr(anp, "read_anp_sheet", lambda path: source)
    result = anp.parse_monthly(Path("fake.xlsx"))
    assert result["data"].dt.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-02-01"]
    assert result["revenda"].tolist() == [4.0, 4.2]
    assert result["distribuicao"].tolist() == [3.6, 3.8]


def test_parse_monthly_allows_missing_distribution(monkeypatch):
    source = pd.DataFrame(
        {
            "PRODUTO": ["DIESEL S-10"],
            "MES": ["01/01/2020"],
            "PREÇO MÉDIO REVENDA": [4.0],
        }
    )
    monkeypatch.setattr(anp, "read_anp_sheet", lambda path: source)
    result = anp.parse_monthly(Path("fake.xlsx"))
    assert result["distribuicao"].isna().all()


def test_parse_weekly_contract(monkeypatch):
    source = pd.DataFrame(
        {
            "PRODUTO": ["OLEO DIESEL S10", "OLEO DIESEL S10"],
            "DATA INICIAL": ["05/01/2020", "05/01/2020"],
            "DATA FINAL": ["11/01/2020", "11/01/2020"],
            "PREÇO MÉDIO REVENDA": [4.0, 8.0],
        }
    )
    monkeypatch.setattr(anp, "read_anp_sheet", lambda path: source)
    result = anp.parse_weekly(Path("fake.xlsx"))
    assert len(result) == 1
    assert result.loc[0, "data"] == pd.Timestamp("2020-01-05")
    assert result.loc[0, "data_final"] == pd.Timestamp("2020-01-11")
    assert pd.isna(result.loc[0, "distribuicao"])

