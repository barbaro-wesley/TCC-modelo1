"""Sparse online Kernel Recursive Least Squares.

The dictionary insertion follows the block inverse and coefficient recursions
reported for ePL-KRLS.  Coherent samples update the existing kernel basis with
an RLS step; bounded replay is used only to rebuild safely after replacement,
kernel-width changes, or rule fusion -- cases not specified by the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .kernels import rbf_features
from .utils import as_finite_target, as_finite_vector


@dataclass(frozen=True)
class KRLSUpdate:
    """Description of one consequent update."""

    prediction: float
    error: float
    action: Literal["initialized", "updated", "inserted", "replaced"]
    dictionary_size: int
    novelty: float


class SparseKRLS:
    """Bounded sparse KRLS consequent with Gaussian kernels.

    Parameters
    ----------
    sigma:
        Initial Gaussian width ``nu``.
    regularization:
        Positive ridge value ``lambda``.
    novelty_factor:
        Implements ``delta = novelty_factor * nu_nearest``; the dissertation
        uses 0.1.
    max_dictionary_size:
        Hard memory bound per fuzzy rule.
    replacement_strategy:
        Policy when a novel, useful sample arrives at the hard bound.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.5,
        regularization: float = 1e-4,
        novelty_factor: float = 0.1,
        max_dictionary_size: int = 40,
        replacement_strategy: Literal["oldest", "least_used", "none"] = "oldest",
        forgetting_factor: float = 1.0,
        usage_decay: float = 1.0,
        min_width: float = 1e-3,
        max_width: float = 10.0,
        replay_capacity: int = 256,
        min_denominator: float = 1e-10,
    ) -> None:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(novelty_factor) or novelty_factor < 0:
            raise ValueError("novelty_factor must be finite and non-negative")
        if max_dictionary_size < 1 or replay_capacity < 1:
            raise ValueError("dictionary and replay capacities must be positive")
        if replacement_strategy not in {"oldest", "least_used", "none"}:
            raise ValueError("unsupported replacement_strategy")
        if not 0 < forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if not 0 < usage_decay <= 1:
            raise ValueError("usage_decay must be in (0, 1]")
        if not 0 < min_width <= max_width:
            raise ValueError("kernel width bounds are invalid")

        self.sigma = float(sigma)
        self.regularization = float(regularization)
        self.novelty_factor = float(novelty_factor)
        self.max_dictionary_size = int(max_dictionary_size)
        self.replacement_strategy = replacement_strategy
        self.forgetting_factor = float(forgetting_factor)
        self.usage_decay = float(usage_decay)
        self.min_width = float(min_width)
        self.max_width = float(max_width)
        self.replay_capacity = int(replay_capacity)
        self.min_denominator = float(min_denominator)
        self.reset()

    def reset(self) -> None:
        self.dictionary_: NDArray[np.float64] = np.empty((0, 0), dtype=float)
        self.widths_: NDArray[np.float64] = np.empty(0, dtype=float)
        self.coefficients_: NDArray[np.float64] = np.empty(0, dtype=float)
        self.Q_: NDArray[np.float64] = np.empty((0, 0), dtype=float)
        self.P_: NDArray[np.float64] = np.empty((0, 0), dtype=float)
        self.width_covariance_: NDArray[np.float64] = np.empty((0, 0), dtype=float)
        self.created_at_: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.usage_: NDArray[np.float64] = np.empty(0, dtype=float)
        self.n_updates_ = 0
        self.n_insertions_ = 0
        self.n_replacements_ = 0
        self._replay_x: list[NDArray[np.float64]] = []
        self._replay_y: list[float] = []

    @property
    def dictionary_size(self) -> int:
        return int(self.dictionary_.shape[0])

    @property
    def n_features_in_(self) -> int | None:
        return None if self.dictionary_size == 0 else int(self.dictionary_.shape[1])

    def _validate_x(self, x: ArrayLike) -> NDArray[np.float64]:
        return as_finite_vector(x, expected_features=self.n_features_in_)

    def _features(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.dictionary_size == 0:
            return np.empty(0, dtype=float)
        return rbf_features(
            x,
            self.dictionary_,
            self.widths_,
            min_width=self.min_width,
        )

    def predict(self, x: ArrayLike) -> float:
        """Predict one sample without changing any state."""

        vector = self._validate_x(x)
        if self.dictionary_size == 0:
            return 0.0
        prediction = float(np.dot(self.coefficients_, self._features(vector)))
        return prediction if np.isfinite(prediction) else 0.0

    def _remember(self, x: NDArray[np.float64], y: float) -> None:
        self._replay_x.append(x.copy())
        self._replay_y.append(y)
        overflow = len(self._replay_y) - self.replay_capacity
        if overflow > 0:
            del self._replay_x[:overflow]
            del self._replay_y[:overflow]

    def _initialize(self, x: NDArray[np.float64], y: float) -> None:
        q = 1.0 / (self.regularization + 1.0)
        self.dictionary_ = x.reshape(1, -1).copy()
        self.widths_ = np.array([self.sigma], dtype=float)
        self.coefficients_ = np.array([q * y], dtype=float)
        self.Q_ = np.array([[q]], dtype=float)
        self.P_ = np.array([[q]], dtype=float)
        self.width_covariance_ = np.eye(1, dtype=float) * 0.05
        self.created_at_ = np.array([self.n_updates_], dtype=np.int64)
        self.usage_ = np.array([1.0], dtype=float)
        self.n_insertions_ += 1

    def _insertion_candidate(
        self,
        x: NDArray[np.float64],
        y: float,
        prediction: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        g = self._features(x)
        z = self.Q_ @ g
        residual_power = self.regularization + 1.0 - float(z @ g)
        if not np.isfinite(residual_power) or residual_power <= self.min_denominator:
            raise np.linalg.LinAlgError("KRLS insertion denominator is singular")
        error = y - prediction
        n = self.dictionary_size
        q_new = np.empty((n + 1, n + 1), dtype=float)
        q_new[:n, :n] = self.Q_ + np.outer(z, z) / residual_power
        q_new[:n, n] = -z / residual_power
        q_new[n, :n] = -z / residual_power
        q_new[n, n] = 1.0 / residual_power
        coefficients = np.empty(n + 1, dtype=float)
        coefficients[:n] = self.coefficients_ - z * error / residual_power
        coefficients[n] = error / residual_power
        if not np.all(np.isfinite(q_new)) or not np.all(np.isfinite(coefficients)):
            raise np.linalg.LinAlgError("KRLS insertion produced non-finite values")
        return q_new, coefficients, residual_power

    def _candidate_prediction(
        self,
        x: NDArray[np.float64],
        coefficients: NDArray[np.float64],
    ) -> float:
        candidate_dictionary = np.vstack((self.dictionary_, x))
        candidate_widths = np.append(self.widths_, self.sigma)
        features = rbf_features(x, candidate_dictionary, candidate_widths)
        return float(coefficients @ features)

    def _insert(
        self,
        x: NDArray[np.float64],
        q_new: NDArray[np.float64],
        coefficients: NDArray[np.float64],
    ) -> None:
        n = self.dictionary_size
        self.dictionary_ = np.vstack((self.dictionary_, x))
        self.widths_ = np.append(self.widths_, self.sigma)
        self.Q_ = q_new
        self.coefficients_ = coefficients
        p_new = np.zeros((n + 1, n + 1), dtype=float)
        p_new[:n, :n] = self.P_
        p_new[n, n] = 1.0 / (self.regularization + 1.0)
        self.P_ = p_new
        width_covariance = np.zeros((n + 1, n + 1), dtype=float)
        width_covariance[:n, :n] = self.width_covariance_
        width_covariance[n, n] = 0.05
        self.width_covariance_ = width_covariance
        self.created_at_ = np.append(self.created_at_, self.n_updates_)
        self.usage_ = np.append(self.usage_, 1.0)
        self.n_insertions_ += 1

    def _fixed_dictionary_update(
        self,
        features: NDArray[np.float64],
        error: float,
    ) -> None:
        p_feature = self.P_ @ features
        denominator = self.forgetting_factor + float(features @ p_feature)
        if not np.isfinite(denominator) or denominator <= self.min_denominator:
            self._rebuild_from_replay()
            return
        gain = p_feature / denominator
        coefficients = self.coefficients_ + gain * error
        p_new = (self.P_ - np.outer(gain, p_feature)) / self.forgetting_factor
        p_new = 0.5 * (p_new + p_new.T)
        if np.all(np.isfinite(coefficients)) and np.all(np.isfinite(p_new)):
            self.coefficients_ = coefficients
            self.P_ = p_new
        else:
            self._rebuild_from_replay()

    def _replacement_index(self) -> int:
        if self.replacement_strategy == "oldest":
            return int(np.argmin(self.created_at_))
        if self.replacement_strategy == "least_used":
            return int(np.lexsort((self.created_at_, self.usage_))[0])
        raise RuntimeError("replacement was requested with strategy='none'")

    def _replace(self, x: NDArray[np.float64]) -> None:
        index = self._replacement_index()
        self.dictionary_[index] = x
        self.widths_[index] = self.sigma
        self.created_at_[index] = self.n_updates_
        self.usage_[index] = 1
        self.n_replacements_ += 1
        self._rebuild_from_replay()

    def _symmetric_dictionary_gram(self) -> NDArray[np.float64]:
        differences = self.dictionary_[:, None, :] - self.dictionary_[None, :, :]
        distance_squared = np.sum(differences**2, axis=2)
        pair_width = np.sqrt(self.widths_[:, None] * self.widths_[None, :])
        return np.exp(-distance_squared / (2.0 * np.maximum(pair_width, self.min_width) ** 2))

    @staticmethod
    def _stable_inverse(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(matrix, rcond=1e-10)
        return 0.5 * (inverse + inverse.T)

    def _rebuild_from_replay(self) -> None:
        if self.dictionary_size == 0:
            return
        gram = self._symmetric_dictionary_gram()
        self.Q_ = self._stable_inverse(
            gram + self.regularization * np.eye(self.dictionary_size)
        )
        if not self._replay_y:
            self.P_ = np.eye(self.dictionary_size) / self.regularization
            self.coefficients_ = np.zeros(self.dictionary_size)
            return
        design = np.vstack([self._features(x) for x in self._replay_x])
        targets = np.asarray(self._replay_y, dtype=float)
        system = design.T @ design + self.regularization * np.eye(self.dictionary_size)
        self.P_ = self._stable_inverse(system)
        self.coefficients_ = self.P_ @ design.T @ targets
        if not np.all(np.isfinite(self.coefficients_)):
            self.coefficients_ = np.linalg.pinv(design, rcond=1e-10) @ targets

    def update(
        self,
        x: ArrayLike,
        y: float,
        *,
        allow_growth: bool = True,
        require_error_improvement: bool = True,
    ) -> KRLSUpdate:
        """Predict, then update from one labelled sample."""

        vector = self._validate_x(x)
        target = as_finite_target(y)
        prediction = self.predict(vector)
        error = target - prediction
        self.n_updates_ += 1
        self._remember(vector, target)

        if self.dictionary_size == 0:
            self._initialize(vector, target)
            return KRLSUpdate(prediction, error, "initialized", 1, np.inf)

        features = self._features(vector)
        distances = np.linalg.norm(self.dictionary_ - vector[None, :], axis=1)
        nearest = int(np.argmin(distances))
        novelty = float(distances[nearest])
        self.usage_ *= self.usage_decay
        self.usage_[nearest] += 1.0
        threshold = self.novelty_factor * max(self.widths_[nearest], self.min_width)

        if allow_growth and novelty >= threshold:
            try:
                q_new, coefficients, _ = self._insertion_candidate(vector, target, prediction)
                candidate_prediction = self._candidate_prediction(vector, coefficients)
                improves = abs(target - candidate_prediction) < abs(error) - 1e-12
                useful = improves or not require_error_improvement
            except np.linalg.LinAlgError:
                useful = False
                q_new = np.empty((0, 0))
                coefficients = np.empty(0)
            if useful and self.dictionary_size < self.max_dictionary_size:
                self._insert(vector, q_new, coefficients)
                return KRLSUpdate(prediction, error, "inserted", self.dictionary_size, novelty)
            if useful and self.replacement_strategy != "none":
                self._replace(vector)
                return KRLSUpdate(prediction, error, "replaced", self.dictionary_size, novelty)

        self._fixed_dictionary_update(features, error)
        return KRLSUpdate(prediction, error, "updated", self.dictionary_size, novelty)

    def adapt_widths(
        self,
        x: ArrayLike,
        error: float,
        *,
        activation: float = 1.0,
        max_relative_change: float = 0.1,
    ) -> bool:
        """Apply the recursive gradient width update from Eq. 15.

        The derivative uses the analytic positive derivative of the Gaussian
        with respect to its width.  Every change is clipped and followed by a
        numerically stable rebuild because changing widths invalidates ``Q``.
        """

        vector = self._validate_x(x)
        if self.dictionary_size == 0 or not np.isfinite(error + activation):
            return False
        features = self._features(vector)
        distance_squared = np.sum((self.dictionary_ - vector[None, :]) ** 2, axis=1)
        safe_widths = np.maximum(self.widths_, self.min_width)
        gradient = (
            float(activation)
            * self.coefficients_
            * features
            * distance_squared
            / safe_widths**3
        )
        covariance_gradient = self.width_covariance_ @ gradient
        denominator = 1.0 + float(gradient @ covariance_gradient)
        if not np.isfinite(denominator) or denominator <= self.min_denominator:
            return False
        covariance = self.width_covariance_ - np.outer(
            covariance_gradient, covariance_gradient
        ) / denominator
        delta = covariance @ gradient * float(error)
        limit = max_relative_change * safe_widths
        delta = np.clip(delta, -limit, limit)
        widths = np.clip(self.widths_ + delta, self.min_width, self.max_width)
        if not np.all(np.isfinite(widths)) or np.allclose(widths, self.widths_):
            return False
        self.widths_ = widths
        self.width_covariance_ = 0.5 * (covariance + covariance.T)
        self._rebuild_from_replay()
        return True

    def absorb(self, other: "SparseKRLS") -> None:
        """Merge another rule consequent using bounded replay and rebuilding."""

        if other.dictionary_size == 0:
            return
        if self.dictionary_size == 0:
            self.dictionary_ = other.dictionary_.copy()
            self.widths_ = other.widths_.copy()
        elif self.dictionary_.shape[1] != other.dictionary_.shape[1]:
            raise ValueError("cannot merge KRLS models with different feature counts")
        else:
            slots = max(0, self.max_dictionary_size - self.dictionary_size)
            if slots:
                self.dictionary_ = np.vstack((self.dictionary_, other.dictionary_[:slots]))
                self.widths_ = np.append(self.widths_, other.widths_[:slots])
        self._replay_x = (self._replay_x + [x.copy() for x in other._replay_x])[-self.replay_capacity :]
        self._replay_y = (self._replay_y + list(other._replay_y))[-self.replay_capacity :]
        n = self.dictionary_size
        self.created_at_ = np.arange(n, dtype=np.int64)
        self.usage_ = np.ones(n, dtype=float)
        self.width_covariance_ = np.eye(n) * 0.05
        self._rebuild_from_replay()

    def inspect(self) -> dict[str, object]:
        """Return a copy-safe, serializable model summary."""

        return {
            "dictionary_size": self.dictionary_size,
            "dictionary": self.dictionary_.copy().tolist(),
            "widths": self.widths_.copy().tolist(),
            "coefficients": self.coefficients_.copy().tolist(),
            "updates": self.n_updates_,
            "insertions": self.n_insertions_,
            "replacements": self.n_replacements_,
            "replacement_rate": float(
                self.n_replacements_ / max(self.n_updates_, 1)
            ),
            "usage": self.usage_.copy().tolist(),
        }
