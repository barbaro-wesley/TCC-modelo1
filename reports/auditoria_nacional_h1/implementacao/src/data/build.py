from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from data.anp import parse_monthly, parse_weekly
from vsepl_krls.paper import TABLE1_S10

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

ANP_GAP_START = pd.Timestamp("2020-08-18")
ANP_GAP_END = pd.Timestamp("2020-10-17")
DIST_END = pd.Timestamp("2020-08-17")


def load_monthly_s10() -> pd.DataFrame:
    old = parse_monthly(RAW / "anp_mensal_2001_2012.xlsx")
    new = parse_monthly(RAW / "anp_mensal_desde_2013.xlsx")
    df = pd.concat([old, new], ignore_index=True)
    df = df.drop_duplicates("data").sort_values("data").reset_index(drop=True)
    return df


def load_weekly_s10() -> pd.DataFrame:
    df = parse_weekly(RAW / "anp_semanal_desde_2013.xlsx")
    df["in_anp_gap"] = (df["data"] >= ANP_GAP_START) & (df["data"] <= ANP_GAP_END)
    df["distribuicao_disponivel"] = df["data"] <= DIST_END
    df.loc[~df["distribuicao_disponivel"], "distribuicao"] = np.nan
    return df


def paper_window_monthly(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp("2012-12-01")
    end = pd.Timestamp("2020-05-31")
    out = df[(df["data"] >= start) & (df["data"] <= end)].copy()
    return out.reset_index(drop=True)


def table1_check(series: pd.Series, expected: dict = TABLE1_S10, atol: float = 0.02) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    obs = {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q1": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q3": float(s.quantile(0.75)),
        "max": float(s.max()),
    }
    ok = True
    diffs = {}
    for k, exp in expected.items():
        if k == "n":
            diffs[k] = obs[k] - exp
            ok = ok and obs[k] == exp
        else:
            diffs[k] = obs[k] - exp
            ok = ok and abs(obs[k] - exp) <= atol
    return {"ok": ok, "observed": obs, "expected": expected, "diffs": diffs}


def load_brent() -> pd.Series:
    raw = json.loads((RAW / "ipeadata_brent.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["data"] = pd.to_datetime(df["VALDATA"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["brent"] = pd.to_numeric(df["VALVALOR"], errors="coerce")
    df = df.dropna(subset=["data", "brent"]).sort_values("data")
    return df.set_index("data")["brent"]


def load_usdbrl() -> pd.Series:
    path = RAW / "ipeadata_usdbrl.json"
    if not path.exists():
        path = RAW / "bcb_ptax.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(raw)
        df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        df["usdbrl"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["data", "usdbrl"]).sort_values("data")
        return df.set_index("data")["usdbrl"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["data"] = pd.to_datetime(df["VALDATA"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["usdbrl"] = pd.to_numeric(df["VALVALOR"], errors="coerce")
    df = df.dropna(subset=["data", "usdbrl"]).sort_values("data")
    return df.set_index("data")["usdbrl"]


def load_ulsd() -> pd.Series:
    path = RAW / "stooq_ulsd.csv"
    if not path.exists():
        return pd.Series(dtype=float, name="ulsd")
    df = pd.read_csv(StringIO(path.read_text(encoding="utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        return pd.Series(dtype=float, name="ulsd")
    df["data"] = pd.to_datetime(df["date"], errors="coerce")
    df["ulsd"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["data", "ulsd"]).sort_values("data")
    return df.set_index("data")["ulsd"]


def resample_to_week(series: pd.Series, week_index: pd.DatetimeIndex) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    daily = s.resample("D").ffill()
    aligned = daily.reindex(week_index, method="ffill")
    return aligned


def petrobras_proxy(weekly: pd.DataFrame) -> pd.DataFrame:
    """Fallback: jump in pump price plus deviation from Brent-BRL parity."""
    out = weekly.copy()
    r = out["revenda"].astype(float)
    dlog = np.log(r).diff()
    sigma = dlog.rolling(12, min_periods=4).std()
    jump = (dlog.abs() > (2.5 * sigma.clip(lower=1e-4))).astype(float)
    if "brent_brl" in out.columns:
        parity = out["revenda"] / out["brent_brl"].replace(0, np.nan)
        z = (parity - parity.rolling(12, min_periods=4).mean()) / parity.rolling(12, min_periods=4).std()
        out["paridade_z"] = z
        out["petrobras_reajuste"] = ((jump == 1) | (z.abs() > 2.0)).astype(float)
    else:
        out["paridade_z"] = np.nan
        out["petrobras_reajuste"] = jump
    out["revenda_jump"] = jump
    return out


def build_weekly_features(weekly: pd.DataFrame, brent: pd.Series, usdbrl: pd.Series, ulsd: pd.Series) -> pd.DataFrame:
    df = weekly.copy().sort_values("data").reset_index(drop=True)
    idx = pd.DatetimeIndex(df["data"])
    df["brent"] = resample_to_week(brent, idx).to_numpy()
    df["usdbrl"] = resample_to_week(usdbrl, idx).to_numpy()
    if len(ulsd):
        df["ulsd"] = resample_to_week(ulsd, idx).to_numpy()
    else:
        df["ulsd"] = np.nan
    df["brent_brl"] = df["brent"] * df["usdbrl"]
    df = petrobras_proxy(df)
    r = df["revenda"].astype(float)
    dlog = np.log(r).diff()
    for lag in (1, 2, 4, 8, 12):
        df[f"revenda_l{lag}"] = r.shift(lag)
        df[f"brent_l{lag}"] = df["brent"].shift(lag)
        df[f"usdbrl_l{lag}"] = df["usdbrl"].shift(lag)
        df[f"brent_brl_l{lag}"] = df["brent_brl"].shift(lag)
        df[f"ulsd_l{lag}"] = df["ulsd"].shift(lag)
    for w in (4, 8, 12):
        df[f"revenda_ma{w}"] = r.shift(1).rolling(w, min_periods=w).mean()
        df[f"vol{w}"] = dlog.shift(1).rolling(w, min_periods=max(3, w // 2)).std()
    df["petrobras_reajuste_l1"] = df["petrobras_reajuste"].shift(1)
    df["paridade_z_l1"] = df["paridade_z"].shift(1)
    df["distribuicao_l1"] = df["distribuicao"].shift(1)
    return df


def leak_check(df: pd.DataFrame, feature_cols: list, date_col: str = "data") -> list:
    """Return feature names that appear contemporaneous (not lagged) with the target week."""
    leaks = []
    forbidden = {
        "revenda",
        "distribuicao",
        "brent",
        "usdbrl",
        "ulsd",
        "brent_brl",
        "petrobras_reajuste",
        "paridade_z",
        "revenda_jump",
    }
    for c in feature_cols:
        if c in forbidden:
            leaks.append(c)
        if c.endswith("_l0"):
            leaks.append(c)
    return sorted(set(leaks))


def save_processed() -> dict:
    PROC.mkdir(parents=True, exist_ok=True)
    monthly = load_monthly_s10()
    weekly = load_weekly_s10()
    brent = load_brent()
    fx = load_usdbrl()
    ulsd = load_ulsd()
    paper = paper_window_monthly(monthly)
    weekly_feat = build_weekly_features(weekly, brent, fx, ulsd)
    monthly.to_csv(PROC / "mensal_s10.csv", index=False)
    weekly.to_csv(PROC / "semanal_s10.csv", index=False)
    paper.to_csv(PROC / "mensal_s10_artigo.csv", index=False)
    weekly_feat.to_csv(PROC / "semanal_s10_features.csv", index=False)
    gate = table1_check(paper["revenda"])
    (PROC / "table1_gate.json").write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"monthly": monthly, "weekly": weekly, "paper": paper, "features": weekly_feat, "gate": gate}
