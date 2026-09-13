"""Intervalo preditivo condicional ao regime, aplicavel a qualquer previsor pontual.

Problema
--------
O bundle 1.1 calibra o intervalo com quantis de uma janela movel de residuos.
Como a largura nao depende do estado do mercado, ela e larga demais nas semanas
paradas (dois tercos da amostra) e estreita demais nas semanas de choque.  No
holdout de 104 semanas isso aparece como cobertura de 92,3% para um nominal de
80%: o intervalo nao esta ``conservador'', esta desinformativo.

Abordagem
---------
A escala do erro e modelada explicitamente a partir de duas variaveis causais
disponiveis na origem — a volatilidade recente da propria serie e o tamanho do
ultimo movimento do custo do insumo em R$/L::

    sigma(T) = c0 + c1*volatilidade(T) + c2*|variacao de custo|(T)

Os residuos padronizados por ``sigma`` sao entao usados para extrair os quantis
empiricos.  Isso preserva a calibracao global e faz a largura responder ao
regime.  A alternativa por mistura de dois regimes foi testada e rejeitada nos
folds de desenvolvimento (cobertura 64,7% contra nominal 80%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["ConditionalIntervalCalibrator", "IntervalBand"]

_SCALE_FEATURES: tuple[str, ...] = ("volatility", "abs_cost_move")


@dataclass(frozen=True)
class IntervalBand:
    lower: float
    upper: float
    sigma: float
    nominal: float

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)


@dataclass
class ConditionalIntervalCalibrator:
    """Calibra ``sigma`` condicional e quantis padronizados a partir de residuos."""

    nominal: float = 0.80
    min_samples: int = 60
    scale_features: tuple[str, ...] = _SCALE_FEATURES
    coefficients_: NDArray[np.float64] | None = None
    quantiles_: tuple[float, float] | None = None
    floor_: float = 1e-4

    def __post_init__(self) -> None:
        if not 0.5 <= self.nominal < 1.0:
            raise ValueError("nominal must be in [0.5, 1)")
        if self.min_samples < 20:
            raise ValueError("min_samples must be at least 20")

    def fit(self, residuals: Sequence[float], scale_frame: pd.DataFrame) -> "ConditionalIntervalCalibrator":
        residual = np.asarray(residuals, dtype=float)
        raw = scale_frame[list(self.scale_features)].to_numpy(float)
        if raw.shape[0] != residual.shape[0]:
            raise ValueError("residuals and scale features must have the same length")
        usable = np.isfinite(residual) & np.isfinite(raw).all(axis=1)
        if int(usable.sum()) < self.min_samples:
            raise ValueError("not enough usable residuals to calibrate")
        residual, raw = residual[usable], raw[usable]
        absolute = np.abs(residual)
        self.floor_ = max(float(np.median(absolute)) * 0.25, 1e-4)
        design = np.column_stack([np.ones(len(raw)), raw])
        coefficients, *_ = np.linalg.lstsq(design, absolute, rcond=None)
        if not np.all(np.isfinite(coefficients)):
            self.coefficients_ = None
            standardized = residual / self.floor_
        else:
            self.coefficients_ = coefficients
            predicted = np.maximum(design @ coefficients, self.floor_)
            standardized = residual / predicted
        tail = (1.0 - self.nominal) / 2.0
        clean = standardized[np.isfinite(standardized)]
        low, high = np.quantile(clean, [tail, 1.0 - tail])
        self.quantiles_ = (float(low), float(high))
        return self

    def sigma(self, row: pd.Series) -> float:
        if self.coefficients_ is None:
            return self.floor_
        values = np.array([1.0] + [float(row.get(name, np.nan)) for name in self.scale_features])
        if not np.all(np.isfinite(values)):
            return self.floor_
        return max(float(values @ self.coefficients_), self.floor_)

    def band(self, point: float, row: pd.Series) -> IntervalBand:
        if self.quantiles_ is None:
            raise RuntimeError("calibrator is not fitted")
        scale = self.sigma(row)
        low, high = self.quantiles_
        return IntervalBand(
            lower=float(point + low * scale),
            upper=float(point + high * scale),
            sigma=scale,
            nominal=self.nominal,
        )

    def summary(self) -> dict[str, object]:
        return {
            "fitted": self.quantiles_ is not None,
            "nominal": self.nominal,
            "standardized_quantiles": (
                None if self.quantiles_ is None else [round(v, 6) for v in self.quantiles_]
            ),
            "sigma_coefficients": (
                None
                if self.coefficients_ is None
                else {
                    name: round(float(value), 8)
                    for name, value in zip(("intercept",) + self.scale_features, self.coefficients_)
                }
            ),
            "floor": round(float(self.floor_), 8),
        }
