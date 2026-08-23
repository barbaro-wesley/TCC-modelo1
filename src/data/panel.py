"""Painel semanal compartilhado entre a avaliacao walk-forward e a producao.

Ficava dentro de scripts/03_semanal.py. Foi extraido para ca porque o job
semanal precisa montar exatamente o mesmo painel na hora de gerar a previsao
publicada — se as duas montagens divergirem, o modelo escolhido pelo RMSE nao
e o modelo que de fato produz o numero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [
    "revenda_l1", "revenda_l2", "revenda_l4", "revenda_l8", "revenda_l12",
    "revenda_ma4", "revenda_ma8", "revenda_ma12",
    "vol4", "vol12",
    "brent_l1", "brent_l4", "brent_brl_l1", "brent_brl_l4",
    "usdbrl_l1", "usdbrl_l4",
    "ulsd_l1", "ulsd_l4",
    "petrobras_reajuste_l1", "paridade_z_l1",
]

ARIMAX_COLS = ["brent_l1", "usdbrl_l1"]

GAP_START = pd.Timestamp("2020-08-18")
GAP_END = pd.Timestamp("2020-10-17")


def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROC / "semanal_s10_features.csv", parse_dates=["data"])
    return df.sort_values("data").reset_index(drop=True)


def load_panel(horizon: int) -> pd.DataFrame:
    """Painel de treino/avaliacao para um horizonte: y = preco h semanas a frente."""
    df = load_features()
    df["y"] = df["revenda"].shift(-horizon)
    df["y_prev"] = df["revenda"]
    future_date = df["data"].shift(-horizon)
    gap_mask = (future_date >= GAP_START) & (future_date <= GAP_END)
    df = df.loc[~gap_mask].copy()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns and float(df[c].notna().mean()) > 0.8]
    df[feat_cols] = df[feat_cols].ffill()
    keep = feat_cols + ["y", "y_prev", "revenda", "data"]
    out = df[keep].dropna(subset=feat_cols + ["y"]).reset_index(drop=True)
    out.attrs["feat_cols"] = feat_cols
    return out


def scale_frozen(X: np.ndarray, n_min: int):
    """Min-max congelado nas primeiras n_min linhas (nao olha o futuro)."""
    lo = np.nanmin(X[:n_min], axis=0)
    hi = np.nanmax(X[:n_min], axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    Xs = np.clip((X - lo) / span, -0.25, 1.25)
    return Xs, lo, span


def latest_features(feat_cols: list) -> Tuple[np.ndarray, pd.Timestamp, float]:
    """Ultima linha observada: features conhecidas, alvo ainda desconhecido.

    E a linha que load_panel descarta (y e NaN) e justamente a que a previsao
    de producao precisa.
    """
    full = load_features()
    full[feat_cols] = full[feat_cols].ffill().bfill()
    full = full.dropna(subset=feat_cols).sort_values("data")
    if full.empty:
        raise ValueError("Nenhuma linha com features completas em semanal_s10_features.csv")
    x_last = full[feat_cols].to_numpy(float)[-1]
    ultima_data = pd.Timestamp(full["data"].iloc[-1])
    ultimo_preco = float(full["revenda"].iloc[-1])
    return x_last, ultima_data, ultimo_preco
