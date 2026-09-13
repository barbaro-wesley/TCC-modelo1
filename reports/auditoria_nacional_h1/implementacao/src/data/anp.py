from __future__ import annotations

from pathlib import Path
import unicodedata

import pandas as pd

ANP_BASE = (
    "https://www.gov.br/anp/pt-br/assuntos/precos-e-defesa-da-concorrencia/"
    "precos/precos-revenda-e-de-distribuicao-combustiveis"
)

URLS = {
    "mensal_2013": f"{ANP_BASE}/shlp/mensal/mensal-brasil-desde-jan2013.xlsx",
    "mensal_2001": f"{ANP_BASE}/shlp/2001-2012/mensal-brasil-2001-a-2012.xlsx",
    "semanal_2013": f"{ANP_BASE}/shlp/semanal/semanal-brasil-desde-2013.xlsx",
}

S10_ALIASES = {
    "OLEO DIESEL S10",
    "ÓLEO DIESEL S10",
    "OLEO DIESEL S-10",
    "ÓLEO DIESEL S-10",
    "DIESEL S10",
    "DIESEL S-10",
}

def _norm(s) -> str:
    t = str(s).strip().upper()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("Ç", "C")
    return t


def _find_header_row(df: pd.DataFrame) -> int:
    keys = {"MES", "MÊS", "DATA INICIAL", "DATAINICIAL"}
    for i, row in df.iterrows():
        vals = {_norm(v) for v in row.tolist() if pd.notna(v)}
        if vals & keys:
            return int(i)
    raise ValueError("Nao encontrei a linha de cabecalho ANP")


def _col(df: pd.DataFrame, *names: str) -> str:
    lookup = {_norm(c): c for c in df.columns}
    for n in names:
        if _norm(n) in lookup:
            return lookup[_norm(n)]
    raise KeyError(f"Coluna nao encontrada: {names}. Disponiveis: {list(df.columns)}")


def read_anp_sheet(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    hdr = _find_header_row(raw)
    df = raw.iloc[hdr + 1 :].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = df.dropna(how="all")
    return df


def filter_s10(df: pd.DataFrame) -> pd.DataFrame:
    prod = _col(df, "PRODUTO")
    mask = df[prod].map(_norm).isin(S10_ALIASES)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError("Nenhuma linha de Diesel S10 na planilha ANP")
    return out


def parse_monthly(path: Path) -> pd.DataFrame:
    df = filter_s10(read_anp_sheet(path))
    mes = _col(df, "MES", "MÊS")
    rev = _col(df, "PREÇO MÉDIO REVENDA", "PRECO MEDIO REVENDA")
    dist_col = None
    for cand in ("PREÇO MÉDIO DISTRIBUIÇÃO", "PRECO MEDIO DISTRIBUICAO"):
        try:
            dist_col = _col(df, cand)
            break
        except KeyError:
            continue
    out = pd.DataFrame(
        {
            "data": pd.to_datetime(df[mes], dayfirst=True, errors="coerce"),
            "revenda": pd.to_numeric(df[rev], errors="coerce"),
        }
    )
    if dist_col is not None:
        out["distribuicao"] = pd.to_numeric(df[dist_col], errors="coerce")
    else:
        out["distribuicao"] = pd.NA
    out = out.dropna(subset=["data", "revenda"]).sort_values("data")
    return out.drop_duplicates("data").reset_index(drop=True)


def parse_weekly(path: Path) -> pd.DataFrame:
    df = filter_s10(read_anp_sheet(path))
    di = _col(df, "DATA INICIAL")
    df_ = _col(df, "DATA FINAL")
    rev = _col(df, "PREÇO MÉDIO REVENDA", "PRECO MEDIO REVENDA")
    dist_col = None
    for cand in ("PREÇO MÉDIO DISTRIBUIÇÃO", "PRECO MEDIO DISTRIBUICAO"):
        try:
            dist_col = _col(df, cand)
            break
        except KeyError:
            continue
    out = pd.DataFrame(
        {
            "data_inicial": pd.to_datetime(df[di], dayfirst=True, errors="coerce"),
            "data_final": pd.to_datetime(df[df_], dayfirst=True, errors="coerce"),
            "revenda": pd.to_numeric(df[rev], errors="coerce"),
        }
    )
    if dist_col is not None:
        out["distribuicao"] = pd.to_numeric(df[dist_col], errors="coerce")
    else:
        out["distribuicao"] = pd.NA
    out = out.dropna(subset=["data_inicial", "revenda"]).sort_values("data_inicial")
    out["data"] = out["data_inicial"]
    return out.drop_duplicates("data").reset_index(drop=True)
