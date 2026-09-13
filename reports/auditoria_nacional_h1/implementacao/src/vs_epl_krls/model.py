"""Online VS-ePL-KRLS regressor.

The structural equations and update order follow Algorithm 1 of the linked
dissertation.  In particular, arousal and structural thresholds use the beta
value from the previous step; the newly adapted beta becomes active at the next
sample.  Prediction is always computed before the current target is consumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import logging
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .krls import SparseKRLS
from .rule import EvolvingRule
from .utils import RunningTargetStats, as_finite_target, as_finite_vector


@dataclass(frozen=True)
class VSEPLKRLSConfig:
    """Configuration of :class:`VSEPLKRLS`.

    Defaults reproduce the common constants reported by the article where a
    value is unambiguous. Experiment-specific ``error_threshold``,
    ``alpha_vs1`` and ``alpha_vs2`` should be selected on a temporal validation
    split rather than on the test period.
    """

    alpha: float = 0.01
    beta_initial: float = 0.18
    beta_min: float = 1e-4
    beta_max: float = 0.999
    alpha_vs1: float = 0.97
    alpha_vs2: float = 0.84
    beta_recovery_rate: float = 0.0
    error_threshold: float = 0.001
    error_normalization: Literal["none", "running_range", "running_std", "fixed"] = "none"
    error_scale: float = 1.0
    error_scale_epsilon: float = 1e-8
    variable_beta: bool = True

    arousal_threshold: float | None = None
    compatibility_threshold: float | None = None
    merge_threshold: float | None = None
    enable_rule_merging: bool = True
    max_rules: int = 20
    center_update: Literal["paper", "compatibility"] = "paper"
    initial_rule_dispersion: float = 0.05
    min_rule_dispersion: float = 1e-6

    kernel_sigma: float = 0.5
    regularization: float = 1e-4
    novelty_factor: float = 0.1
    max_dictionary_size: int = 40
    replacement_strategy: Literal["oldest", "least_used", "none"] = "oldest"
    forgetting_factor: float = 1.0
    dictionary_usage_decay: float = 1.0
    adapt_kernel_width: bool = True
    min_kernel_width: float = 1e-3
    max_kernel_width: float = 10.0
    max_width_relative_change: float = 0.1
    replay_capacity: int = 256

    input_bounds: tuple[float, float] | None = (0.0, 1.0)
    clip_inputs: bool = False
    initial_prediction: float = 0.0
    log_events: bool = False
    random_state: int | None = None

    def __post_init__(self) -> None:
        unit_values = {
            "alpha": self.alpha,
            "beta_initial": self.beta_initial,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "alpha_vs1": self.alpha_vs1,
            "alpha_vs2": self.alpha_vs2,
        }
        for name, value in unit_values.items():
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and in (0, 1)")
        if self.beta_min > self.beta_initial or self.beta_initial > self.beta_max:
            raise ValueError("beta bounds must contain beta_initial")
        if not np.isfinite(self.beta_recovery_rate) or not 0 <= self.beta_recovery_rate <= 1:
            raise ValueError("beta_recovery_rate must be finite and in [0, 1]")
        for name in ("arousal_threshold", "compatibility_threshold", "merge_threshold"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be None or finite in [0, 1]")
        positive = {
            "error_threshold": self.error_threshold,
            "error_scale": self.error_scale,
            "error_scale_epsilon": self.error_scale_epsilon,
            "initial_rule_dispersion": self.initial_rule_dispersion,
            "min_rule_dispersion": self.min_rule_dispersion,
            "kernel_sigma": self.kernel_sigma,
            "regularization": self.regularization,
            "min_kernel_width": self.min_kernel_width,
            "max_kernel_width": self.max_kernel_width,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.novelty_factor < 0 or not np.isfinite(self.novelty_factor):
            raise ValueError("novelty_factor must be finite and non-negative")
        if self.max_rules < 1 or self.max_dictionary_size < 1 or self.replay_capacity < 1:
            raise ValueError("capacity limits must be positive")
        if not 0 < self.forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if not 0 < self.dictionary_usage_decay <= 1:
            raise ValueError("dictionary_usage_decay must be in (0, 1]")
        if not 0 < self.max_width_relative_change <= 1:
            raise ValueError("max_width_relative_change must be in (0, 1]")
        if self.input_bounds is not None:
            lower, upper = self.input_bounds
            if not np.isfinite(lower + upper) or upper <= lower:
                raise ValueError("input_bounds must be finite and increasing")
        if not np.isfinite(self.initial_prediction):
            raise ValueError("initial_prediction must be finite")


class VSEPLKRLS:
    """Variable Step-Size evolving Participatory Learning with KRLS.

    ``learn_one`` returns the prequential prediction made before learning the
    supplied target.  This makes leakage checks explicit and convenient.
    """

    def __init__(
        self,
        config: VSEPLKRLSConfig | None = None,
        **parameters: object,
    ) -> None:
        if config is not None and parameters:
            raise ValueError("pass either config or keyword parameters, not both")
        if config is None:
            known = {item.name for item in fields(VSEPLKRLSConfig)}
            unknown = sorted(set(parameters) - known)
            if unknown:
                raise TypeError(f"unknown configuration parameters: {unknown}")
            config = VSEPLKRLSConfig(**parameters)  # type: ignore[arg-type]
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.reset()

    def reset(self) -> "VSEPLKRLS":
        """Restore the estimator to its unfitted state."""

        self.rules_: list[EvolvingRule] = []
        self.beta_ = float(self.config.beta_initial)
        self.n_seen_ = 0
        self.n_features_in_: int | None = None
        self.n_rule_creations_ = 0
        self.n_rule_merges_ = 0
        self._next_rule_id = 1
        self._target_stats = RunningTargetStats()
        self.history_: list[dict[str, object]] = []
        self.prequential_predictions_: NDArray[np.float64] = np.empty(0, dtype=float)
        return self

    @property
    def n_rules(self) -> int:
        return len(self.rules_)

    def _prepare_x(self, x: ArrayLike) -> NDArray[np.float64]:
        vector = as_finite_vector(x, expected_features=self.n_features_in_)
        if self.config.input_bounds is not None:
            lower, upper = self.config.input_bounds
            outside = np.any(vector < lower) or np.any(vector > upper)
            if outside and not self.config.clip_inputs:
                raise ValueError(
                    f"x is outside input_bounds={self.config.input_bounds}; "
                    "scale from training data or set clip_inputs=True/input_bounds=None"
                )
            if outside:
                vector = np.clip(vector, lower, upper)
        return vector

    def _new_krls(self) -> SparseKRLS:
        return SparseKRLS(
            sigma=self.config.kernel_sigma,
            regularization=self.config.regularization,
            novelty_factor=self.config.novelty_factor,
            max_dictionary_size=self.config.max_dictionary_size,
            replacement_strategy=self.config.replacement_strategy,
            forgetting_factor=self.config.forgetting_factor,
            usage_decay=self.config.dictionary_usage_decay,
            min_width=self.config.min_kernel_width,
            max_width=self.config.max_kernel_width,
            replay_capacity=self.config.replay_capacity,
        )

    def _create_rule(self, x: NDArray[np.float64], y: float) -> EvolvingRule:
        consequent = self._new_krls()
        consequent.update(x, y)
        rule = EvolvingRule(
            rule_id=self._next_rule_id,
            center=x.copy(),
            dispersion=np.full(x.size, self.config.initial_rule_dispersion),
            arousal=0.0,
            krls=consequent,
            created_at=self.n_seen_,
            last_updated=self.n_seen_,
            activations=1,
            metadata={"creation_sample": self.n_seen_},
        )
        self._next_rule_id += 1
        self.n_rule_creations_ += 1
        self.rules_.append(rule)
        return rule

    def _compatibilities(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray([rule.compatibility(x) for rule in self.rules_], dtype=float)

    @staticmethod
    def _normalized_activations(compatibilities: NDArray[np.float64]) -> NDArray[np.float64]:
        total = float(np.sum(compatibilities))
        if not np.isfinite(total) or total <= 1e-12:
            return np.full(compatibilities.size, 1.0 / compatibilities.size)
        return compatibilities / total

    def _prediction_components(
        self,
        x: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        compatibilities = self._compatibilities(x)
        activations = self._normalized_activations(compatibilities)
        local_predictions = np.asarray(
            [rule.krls.predict(x) for rule in self.rules_],
            dtype=float,
        )
        prediction = float(activations @ local_predictions)
        if not np.isfinite(prediction):
            prediction = float(self.config.initial_prediction)
        return prediction, compatibilities, activations, local_predictions

    def predict_one(self, x: ArrayLike) -> float:
        """Predict one sample without updating the model."""

        vector = self._prepare_x(x)
        if not self.rules_:
            return float(self.config.initial_prediction)
        return self._prediction_components(vector)[0]

    def _normalized_error(self, error: float) -> float:
        mode = self.config.error_normalization
        if mode == "none":
            scale = 1.0
        elif mode == "fixed":
            scale = self.config.error_scale
        elif mode == "running_range":
            scale = self._target_stats.range
        elif mode == "running_std":
            scale = self._target_stats.std
        else:  # protected by the Literal annotation, retained for loaded pickles
            raise ValueError(f"unsupported error normalization: {mode}")
        scale = max(float(scale), self.config.error_scale_epsilon)
        normalized = error / scale
        return float(normalized) if np.isfinite(normalized) else 0.0

    def _adapt_beta(self, normalized_error: float) -> float:
        previous = self.beta_
        if not self.config.variable_beta:
            return previous
        if abs(normalized_error) > self.config.error_threshold:
            candidate = previous / self.config.alpha_vs1
            if self.config.beta_recovery_rate > 0:
                severity = min(
                    max(abs(normalized_error) / self.config.error_threshold - 1.0, 0.0),
                    1.0,
                )
                recovery_floor = self.config.beta_min + (
                    self.config.beta_recovery_rate
                    * severity
                    * (self.config.beta_initial - self.config.beta_min)
                )
                candidate = max(candidate, recovery_floor)
        else:
            candidate = previous * self.config.alpha_vs2
        return float(np.clip(candidate, self.config.beta_min, self.config.beta_max))

    def _thresholds(self, beta_previous: float) -> tuple[float, float]:
        tau = beta_previous if self.config.arousal_threshold is None else self.config.arousal_threshold
        gamma = 1.0 - beta_previous if self.config.merge_threshold is None else self.config.merge_threshold
        return float(np.clip(tau, 0.0, 1.0)), float(np.clip(gamma, 0.0, 1.0))

    def _should_create(
        self,
        compatibilities: NDArray[np.float64],
        tau: float,
    ) -> bool:
        arousal_trigger = min(rule.arousal for rule in self.rules_) > tau
        compatibility_trigger = (
            self.config.compatibility_threshold is not None
            and float(np.max(compatibilities)) < self.config.compatibility_threshold
        )
        return bool(
            self.n_rules < self.config.max_rules
            and (arousal_trigger or compatibility_trigger)
        )

    def _merge_pair(self, first: int, second: int) -> int:
        rule_a, rule_b = self.rules_[first], self.rules_[second]
        if (rule_b.activations, -rule_b.rule_id) > (rule_a.activations, -rule_a.rule_id):
            first, second = second, first
            rule_a, rule_b = rule_b, rule_a
        total = max(rule_a.activations + rule_b.activations, 1)
        weight_a = rule_a.activations / total
        weight_b = rule_b.activations / total
        rule_a.center = np.clip(weight_a * rule_a.center + weight_b * rule_b.center, 0.0, 1.0)
        rule_a.dispersion = np.sqrt(
            np.maximum(
                weight_a * rule_a.dispersion**2 + weight_b * rule_b.dispersion**2,
                self.config.min_rule_dispersion**2,
            )
        )
        rule_a.arousal = float(weight_a * rule_a.arousal + weight_b * rule_b.arousal)
        rule_a.activations = total
        rule_a.merges += 1 + rule_b.merges
        rule_a.krls.absorb(rule_b.krls)
        dropped_id = rule_b.rule_id
        del self.rules_[second]
        self.n_rule_merges_ += 1
        return dropped_id

    def _merge_redundant(self, gamma: float) -> list[int]:
        if not self.config.enable_rule_merging:
            return []
        dropped: list[int] = []
        while len(self.rules_) > 1:
            best: tuple[int, int] | None = None
            best_compatibility = gamma
            for first in range(len(self.rules_)):
                for second in range(first + 1, len(self.rules_)):
                    compatibility = self.rules_[first].pair_compatibility(self.rules_[second])
                    if compatibility > best_compatibility:
                        best = (first, second)
                        best_compatibility = compatibility
            if best is None:
                break
            dropped.append(self._merge_pair(*best))
        return dropped

    def learn_one(self, x: ArrayLike, y: float) -> float:
        """Make a prequential prediction and then learn one labelled sample."""

        vector = self._prepare_x(x)
        target = as_finite_target(y)
        if self.n_features_in_ is None:
            self.n_features_in_ = vector.size
        sample_index = self.n_seen_

        if not self.rules_:
            prediction = float(self.config.initial_prediction)
            error = target - prediction
            normalized_error = self._normalized_error(error)
            created = self._create_rule(vector, target)
            tau, gamma = self._thresholds(self.beta_)
            action = "created_initial"
            winner_id: int | None = created.rule_id
            compatibility_max = 0.0
            beta_previous = self.beta_
            dropped_ids: list[int] = []
        else:
            prediction, compatibilities, activations, local_predictions = self._prediction_components(vector)
            error = target - prediction
            normalized_error = self._normalized_error(error)
            beta_previous = self.beta_

            # Algorithm 1 computes arousal with beta(k-1), before beta(k).
            for rule, compatibility in zip(self.rules_, compatibilities):
                rule.update_arousal(float(compatibility), beta_previous)
            tau, gamma = self._thresholds(beta_previous)
            self.beta_ = self._adapt_beta(normalized_error)
            compatibility_max = float(np.max(compatibilities))

            if self._should_create(compatibilities, tau):
                created = self._create_rule(vector, target)
                winner_id = created.rule_id
                action = "created"
            else:
                winner = int(np.argmax(compatibilities))
                winner_rule = self.rules_[winner]
                winner_id = winner_rule.rule_id
                winner_rule.update_antecedent(
                    vector,
                    alpha=self.config.alpha,
                    compatibility=float(compatibilities[winner]),
                    step=sample_index,
                    mode=self.config.center_update,
                    min_dispersion=self.config.min_rule_dispersion,
                )
                update = winner_rule.krls.update(vector, target)
                action = f"rule_{update.action}"
                if self.config.adapt_kernel_width:
                    local_error = target - float(local_predictions[winner])
                    winner_rule.krls.adapt_widths(
                        vector,
                        local_error,
                        activation=float(activations[winner]),
                        max_relative_change=self.config.max_width_relative_change,
                    )
            dropped_ids = self._merge_redundant(gamma)
            if dropped_ids:
                action += "+merged"

        self._target_stats.update(target)
        self.n_seen_ += 1
        dictionary_sizes = [rule.krls.dictionary_size for rule in self.rules_]
        event: dict[str, object] = {
            "step": sample_index,
            "target": target,
            "prediction": prediction,
            "error": error,
            "normalized_error": normalized_error,
            "beta_previous": beta_previous,
            "beta": self.beta_,
            "tau": tau,
            "gamma": gamma,
            "max_compatibility": compatibility_max,
            "winner_rule_id": winner_id,
            "action": action,
            "merged_rule_ids": dropped_ids,
            "n_rules": self.n_rules,
            "dictionary_sizes": dictionary_sizes,
            "dictionary_size_total": int(sum(dictionary_sizes)),
        }
        self.history_.append(event)
        if self.config.log_events:
            self._logger.info("VS-ePL-KRLS event: %s", event)
        return prediction

    @staticmethod
    def _as_dataset(x: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(x, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("X must be a non-empty finite two-dimensional array")
        return values

    def fit(self, x: ArrayLike, y: ArrayLike) -> "VSEPLKRLS":
        """Process a dataset once in its given sequential order."""

        features = self._as_dataset(x)
        targets = np.asarray(y, dtype=float).ravel()
        if targets.shape != (features.shape[0],) or not np.all(np.isfinite(targets)):
            raise ValueError("y must have one finite scalar per row of X")
        predictions = [self.learn_one(row, target) for row, target in zip(features, targets)]
        self.prequential_predictions_ = np.asarray(predictions, dtype=float)
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        """Return read-only predictions; no rule, beta, or KRLS state changes."""

        features = self._as_dataset(x)
        return np.asarray([self.predict_one(row) for row in features], dtype=float)

    def get_rules(self) -> list[dict[str, object]]:
        """Return copy-safe, interpretable state for every rule."""

        return [rule.inspect() for rule in self.rules_]

    def get_history(self) -> list[dict[str, object]]:
        """Return a shallow copy of every serializable learning event."""

        return [dict(event) for event in self.history_]

    def summary(self) -> dict[str, object]:
        sizes = [rule.krls.dictionary_size for rule in self.rules_]
        replacements = sum(rule.krls.n_replacements_ for rule in self.rules_)
        consequent_updates = sum(rule.krls.n_updates_ for rule in self.rules_)
        history_sizes = [int(event["n_rules"]) for event in self.history_]
        return {
            "n_seen": self.n_seen_,
            "n_rules": self.n_rules,
            "max_rules_observed": max(history_sizes, default=0),
            "rule_creations": self.n_rule_creations_,
            "rule_merges": self.n_rule_merges_,
            "mean_dictionary_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_dictionary_size": max(sizes, default=0),
            "dictionary_replacements": int(replacements),
            "dictionary_replacement_rate": float(
                replacements / max(consequent_updates, 1)
            ),
            "beta": self.beta_,
            "config": asdict(self.config),
        }


class EPLKRLSFixedBeta(VSEPLKRLS):
    """Ablation model with the same implementation and a constant beta."""

    def __init__(
        self,
        config: VSEPLKRLSConfig | None = None,
        **parameters: object,
    ) -> None:
        if config is not None:
            values = asdict(config)
            values["variable_beta"] = False
            config = VSEPLKRLSConfig(**values)
        else:
            parameters["variable_beta"] = False
        super().__init__(config=config, **parameters)
