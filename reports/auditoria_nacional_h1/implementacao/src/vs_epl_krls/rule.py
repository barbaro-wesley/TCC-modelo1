"""Fuzzy rule state for VS-ePL-KRLS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .krls import SparseKRLS
from .utils import as_finite_vector


@dataclass
class EvolvingRule:
    """One evolving antecedent and its independent KRLS consequent."""

    rule_id: int
    center: NDArray[np.float64]
    dispersion: NDArray[np.float64]
    arousal: float
    krls: SparseKRLS
    created_at: int
    last_updated: int
    activations: int = 0
    merges: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
    _mean: NDArray[np.float64] | None = field(default=None, repr=False)
    _m2: NDArray[np.float64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.center = as_finite_vector(self.center, name="center").copy()
        self.dispersion = as_finite_vector(
            self.dispersion,
            name="dispersion",
            expected_features=self.center.size,
        ).copy()
        if np.any(self.dispersion <= 0):
            raise ValueError("dispersion must be positive")
        if not np.isfinite(self.arousal):
            raise ValueError("arousal must be finite")
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))
        if self._mean is None:
            self._mean = self.center.copy()
        if self._m2 is None:
            self._m2 = np.zeros_like(self.center)

    def compatibility(self, x: ArrayLike) -> float:
        """Eq. 3-2: ``rho = 1 - ||x-v|| / m``, clipped for robustness."""

        vector = as_finite_vector(x, expected_features=self.center.size)
        rho = 1.0 - float(np.linalg.norm(vector - self.center)) / self.center.size
        return float(np.clip(rho, 0.0, 1.0))

    def update_arousal(self, compatibility: float, beta: float) -> float:
        """Eq. 3-3 using the beta value from the previous time step."""

        if not np.isfinite(compatibility + beta) or not 0 <= beta <= 1:
            raise ValueError("compatibility and beta must be finite in [0, 1]")
        self.arousal += beta * (1.0 - compatibility - self.arousal)
        self.arousal = float(np.clip(self.arousal, 0.0, 1.0))
        return self.arousal

    def update_antecedent(
        self,
        x: ArrayLike,
        *,
        alpha: float,
        compatibility: float,
        step: int,
        mode: Literal["paper", "compatibility"] = "paper",
        min_dispersion: float = 1e-6,
    ) -> None:
        """Update centre and diagonal dispersion.

        ``paper`` implements Eq. 3-4 literally, with elementwise
        ``center ** (1-arousal)``. ``compatibility`` exposes the common ePL
        alternative because the literal equation freezes zero-valued centre
        components.
        """

        vector = as_finite_vector(x, expected_features=self.center.size)
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite in [0, 1]")
        if mode == "paper":
            base = np.clip(self.center, 0.0, 1.0)
            participation = np.power(base, 1.0 - self.arousal)
        elif mode == "compatibility":
            participation = np.full_like(
                self.center,
                max(float(compatibility), 0.0) ** (1.0 - self.arousal),
            )
        else:
            raise ValueError("mode must be 'paper' or 'compatibility'")
        self.center = np.clip(
            self.center + alpha * participation * (vector - self.center),
            0.0,
            1.0,
        )

        self.activations += 1
        self.last_updated = int(step)
        if self._mean is None or self._m2 is None:
            self._mean = vector.copy()
            self._m2 = np.zeros_like(vector)
        else:
            delta = vector - self._mean
            self._mean += delta / max(self.activations, 1)
            self._m2 += delta * (vector - self._mean)
        if self.activations > 1:
            variance = self._m2 / (self.activations - 1)
            self.dispersion = np.sqrt(np.maximum(variance, min_dispersion**2))

    def pair_compatibility(self, other: "EvolvingRule") -> float:
        """Eq. 3-5: one minus mean absolute centre separation."""

        if self.center.shape != other.center.shape:
            raise ValueError("rules must have centres with equal shape")
        return float(np.clip(1.0 - np.mean(np.abs(self.center - other.center)), 0.0, 1.0))

    def inspect(self) -> dict[str, object]:
        return {
            "id": self.rule_id,
            "center": self.center.copy().tolist(),
            "dispersion": self.dispersion.copy().tolist(),
            "arousal": self.arousal,
            "activations": self.activations,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "merges": self.merges,
            "krls": self.krls.inspect(),
            "metadata": dict(self.metadata),
        }
