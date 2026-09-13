from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data import build


def test_monthly_and_weekly_loaders_merge_and_apply_availability(monkeypatch):
    old = pd.DataFrame({"data": pd.to_datetime(["2012-12-01"]), "revenda": [3.0]})
    new = pd.DataFrame({"data": pd.to_datetime(["2012-12-01", "2013-01-01"]), "revenda": [9.0, 3.1]})
    values = iter([old, new])
    monkeypatch.setattr(build, "parse_monthly", lambda path: next(values))
    monthly = build.load_monthly_s10()
    assert monthly["revenda"].tolist() == [3.0, 3.1]

    weekly_source = pd.DataFrame(
        {
            "data": pd.to_datetime(["2020-08-16", "2020-08-23"]),
            "distribuicao": [4.0, 4.1],
            "revenda": [4.5, 4.6],
        }
    )
    monkeypatch.setattr(build, "parse_weekly", lambda path: weekly_source.copy())
    weekly = build.load_weekly_s10()
    assert weekly["in_anp_gap"].tolist() == [False, True]
    assert weekly["distribuicao_disponivel"].tolist() == [True, False]
    assert np.isnan(weekly.loc[1, "distribuicao"])


def test_paper_window_and_table_gate():
    dates = pd.date_range("2012-11-01", "2020-06-01", freq="MS")
    frame = pd.DataFrame({"data": dates, "revenda": np.arange(len(dates))})
    result = build.paper_window_monthly(frame)
    assert result["data"].min() == pd.Timestamp("2012-12-01")
    assert result["data"].max() == pd.Timestamp("2020-05-01")
    series = pd.Series([1.0, 2.0, 3.0])
    expected = {"n": 3, "mean": 2.0, "std": 1.0, "min": 1.0, "q1": 1.5, "median": 2.0, "q3": 2.5, "max": 3.0}
    assert build.table1_check(series, expected=expected, atol=0.0)["ok"]
    assert not build.table1_check(series, expected={**expected, "mean": 9.0})["ok"]


def test_external_series_loaders(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "RAW", tmp_path)
    ipea = [{"VALDATA": "2020-01-02T00:00:00Z", "VALVALOR": "4.5"}]
    (tmp_path / "ipeadata_brent.json").write_text(json.dumps(ipea), encoding="utf-8")
    (tmp_path / "ipeadata_usdbrl.json").write_text(json.dumps(ipea), encoding="utf-8")
    assert build.load_brent().iloc[0] == 4.5
    assert build.load_usdbrl().iloc[0] == 4.5

    (tmp_path / "ipeadata_usdbrl.json").unlink()
    (tmp_path / "bcb_ptax.json").write_text(
        json.dumps([{"data": "02/01/2020", "valor": "4.1"}]), encoding="utf-8"
    )
    assert build.load_usdbrl().iloc[0] == 4.1


def test_load_ulsd_absent_invalid_and_valid(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "RAW", tmp_path)
    assert build.load_ulsd().empty
    path = tmp_path / "stooq_ulsd.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert build.load_ulsd().empty
    path.write_text("Date,Close\n2020-01-02,2.5\ninvalid,x\n", encoding="utf-8")
    result = build.load_ulsd()
    assert result.name == "ulsd"
    assert result.iloc[0] == 2.5


def test_resample_proxy_and_weekly_features_are_lagged():
    dates = pd.date_range("2020-01-05", periods=16, freq="7D")
    weekly = pd.DataFrame(
        {
            "data": dates,
            "revenda": np.linspace(4.0, 4.6, len(dates)),
            "distribuicao": np.linspace(3.5, 4.1, len(dates)),
        }
    )
    daily_index = pd.date_range("2019-12-20", dates[-1], freq="D")
    brent = pd.Series(np.linspace(60, 70, len(daily_index)), index=daily_index)
    usd = pd.Series(np.linspace(4, 5, len(daily_index)), index=daily_index)
    result = build.build_weekly_features(weekly, brent, usd, pd.Series(dtype=float))
    assert result.loc[1, "revenda_l1"] == weekly.loc[0, "revenda"]
    assert result.loc[4, "revenda_ma4"] == pytest.approx(weekly.loc[:3, "revenda"].mean())
    assert result["ulsd"].isna().all()
    assert {"petrobras_reajuste", "paridade_z", "brent_brl_l1"}.issubset(result.columns)

    no_parity = build.petrobras_proxy(weekly)
    assert no_parity["paridade_z"].isna().all()


def test_leak_check_detects_contemporaneous_and_l0_features():
    leaks = build.leak_check(pd.DataFrame(), ["revenda", "foo_l0", "revenda_l1", "bar"])
    assert leaks == ["foo_l0", "revenda"]


@pytest.mark.integration
def test_save_processed_writes_all_artifacts(monkeypatch, tmp_path):
    monthly = pd.DataFrame({"data": pd.to_datetime(["2012-12-01", "2013-01-01"]), "revenda": [3.0, 3.1]})
    weekly = pd.DataFrame(
        {
            "data": pd.to_datetime(["2020-01-05", "2020-01-12"]),
            "revenda": [4.0, 4.1],
            "distribuicao": [3.5, 3.6],
        }
    )
    monkeypatch.setattr(build, "PROC", tmp_path)
    monkeypatch.setattr(build, "load_monthly_s10", lambda: monthly)
    monkeypatch.setattr(build, "load_weekly_s10", lambda: weekly)
    monkeypatch.setattr(build, "load_brent", lambda: pd.Series([60.0], index=[pd.Timestamp("2020-01-01")]))
    monkeypatch.setattr(build, "load_usdbrl", lambda: pd.Series([4.0], index=[pd.Timestamp("2020-01-01")]))
    monkeypatch.setattr(build, "load_ulsd", lambda: pd.Series(dtype=float))
    monkeypatch.setattr(build, "build_weekly_features", lambda *args: weekly.assign(revenda_l1=[np.nan, 4.0]))
    monkeypatch.setattr(build, "table1_check", lambda series: {"ok": True})
    result = build.save_processed()
    assert result["gate"]["ok"]
    expected = {
        "mensal_s10.csv",
        "semanal_s10.csv",
        "mensal_s10_artigo.csv",
        "semanal_s10_features.csv",
        "table1_gate.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})

