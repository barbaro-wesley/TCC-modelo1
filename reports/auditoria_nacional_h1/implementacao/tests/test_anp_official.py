from __future__ import annotations

import pandas as pd
import pytest

from vs_epl_krls.anp_official import extract_national_s10_observation


def _workbook(path, *, unit="R$/l", start="2026-08-16", end="2026-08-22"):
    frame = pd.DataFrame(
        [
            [start, end, "BRASIL", "GASOLINA COMUM", 1000, "R$/l", 6.2],
            [start, end, "BRASIL", "OLEO DIESEL S10", 3173, unit, 6.89],
        ],
        columns=[
            "DATA INICIAL",
            "DATA FINAL",
            "BRASIL",
            "PRODUTO",
            "NUMERO DE POSTOS PESQUISADOS",
            "UNIDADE DE MEDIDA",
            "PRECO MEDIO REVENDA",
        ],
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="BRASIL", startrow=9, index=False)


def test_extracts_strict_national_s10_observation_with_source_hash(tmp_path):
    path = tmp_path / "official.xlsx"
    _workbook(path)
    result = extract_national_s10_observation(path, source_url="https://example.test/anp.xlsx")
    assert result.week_start == "2026-08-16"
    assert result.week_end == "2026-08-22"
    assert result.price_brl_per_liter == 6.89
    assert result.stations_surveyed == 3173
    assert len(result.source_sha256) == 64
    assert result.source_url == "https://example.test/anp.xlsx"


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"unit": "USD/l"}, "exactly one national Diesel S10"),
        ({"start": "2026-08-17", "end": "2026-08-23"}, "Sunday"),
        ({"end": "2026-08-21"}, "seven calendar days"),
    ],
)
def test_rejects_wrong_unit_or_week_contract(tmp_path, kwargs, message):
    path = tmp_path / "bad.xlsx"
    _workbook(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        extract_national_s10_observation(path)
