from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PaperTargets:
    rmse: float
    mae: float
    ndei: float
    n_rules: int
    n_train: int
    n_obs: int = 90


S10_MONTHLY_H1 = PaperTargets(rmse=0.05953, mae=0.05158, ndei=0.13430, n_rules=1, n_train=72)
S10_MONTHLY_H6 = PaperTargets(rmse=0.10869, mae=0.08366, ndei=0.26269, n_rules=1, n_train=66)
S10_MONTHLY_H12 = PaperTargets(rmse=0.12490, mae=0.10133, ndei=0.33491, n_rules=1, n_train=60)
EPL_KRLS_S10_H1 = PaperTargets(rmse=0.16747, mae=0.11815, ndei=0.37785, n_rules=2, n_train=72)

TABLE1_S10 = {
    "n": 90,
    "mean": 3.114,
    "std": 0.453,
    "min": 2.226,
    "q1": 2.658,
    "median": 3.152,
    "q3": 3.505,
    "max": 3.856,
}


def monthly_params(horizon: int) -> dict:
    if horizon == 1:
        return dict(gamma_bar=0.006, alpha_vs1=0.88, alpha_vs2=0.74)
    if horizon == 6:
        return dict(gamma_bar=0.003, alpha_vs1=0.79, alpha_vs2=0.26)
    if horizon == 12:
        return dict(gamma_bar=0.009, alpha_vs1=0.78, alpha_vs2=0.57)
    raise ValueError(f"horizonte mensal nao coberto pelo artigo: {horizon}")


def make_supervised(revenda: np.ndarray, distribuicao: np.ndarray, horizon: int):
    x, y, idx = [], [], []
    n = len(revenda)
    for t in range(n - horizon):
        if np.isnan(revenda[t]) or np.isnan(distribuicao[t]) or np.isnan(revenda[t + horizon]):
            continue
        x.append([distribuicao[t], revenda[t]])
        y.append(revenda[t + horizon])
        idx.append(t)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(idx, dtype=int)


def scale_inputs_train_only(X_train: np.ndarray, X_test: np.ndarray):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    return Xtr, Xte, scaler


def within_rel(a: float, b: float, tol: float = 0.10) -> bool:
    if b == 0:
        return abs(a - b) <= tol
    return abs(a - b) / abs(b) <= tol


def reproduction_verdict(metrics: Dict[str, float], target: PaperTargets, tol: float = 0.10) -> str:
    ok_err = (
        within_rel(metrics["rmse"], target.rmse, tol)
        and within_rel(metrics["mae"], target.mae, tol)
        and within_rel(metrics["ndei"], target.ndei, tol)
    )
    ok_rules = int(metrics.get("n_rules", -1)) == target.n_rules
    if ok_err and ok_rules:
        return "REPRODUZIDO"
    if ok_err:
        return "PARCIAL (erros ok, regras diferentes)"
    return "NAO REPRODUZIDO"
