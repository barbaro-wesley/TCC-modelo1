import pickle

import numpy as np
import pytest

from vs_epl_krls import VSEPLKRLS, VSEPLKRLSConfig, regression_report


def configured_model():
    return VSEPLKRLS(
        enable_rule_merging=False,
        adapt_kernel_width=False,
        center_update="compatibility",
        kernel_sigma=0.2,
        max_dictionary_size=12,
    )


def test_learn_one_returns_prediction_made_before_current_target():
    model = configured_model()
    model.learn_one([0.0], 0.2)
    before = model.predict_one([0.5])
    returned = model.learn_one([0.5], 0.9)
    assert returned == pytest.approx(before)


def test_predict_is_read_only():
    model = configured_model().fit([[0.0], [0.5], [1.0]], [0.0, 0.5, 1.0])
    history_before = model.get_history()
    rules_before = model.get_rules()
    first = model.predict([[0.2], [0.8]])
    second = model.predict([[0.2], [0.8]])
    assert np.array_equal(first, second)
    assert model.get_history() == history_before
    assert model.get_rules() == rules_before


def test_fit_is_sequential_and_learns_simple_signal():
    x = np.linspace(0, 1, 80).reshape(-1, 1)
    y = x.ravel()
    model = configured_model().fit(x, y)
    metrics = regression_report(y[20:], model.prequential_predictions_[20:])
    assert model.n_seen_ == len(x)
    assert metrics["rmse"] < 0.2


def test_reset_and_pickle_round_trip():
    model = configured_model().fit([[0.0], [1.0]], [0.0, 1.0])
    restored = pickle.loads(pickle.dumps(model))
    assert restored.predict_one([0.5]) == pytest.approx(model.predict_one([0.5]))
    model.reset()
    assert model.n_rules == 0
    assert model.n_seen_ == 0


@pytest.mark.parametrize(
    "x,y",
    [([np.nan], 0.0), ([0.5], np.inf), ([1.5], 0.0)],
)
def test_online_validation(x, y):
    with pytest.raises(ValueError):
        configured_model().learn_one(x, y)


def test_constant_and_repeated_stream_is_stable():
    model = configured_model()
    predictions = [model.learn_one([0.5, 0.5], 0.4) for _ in range(100)]
    assert np.all(np.isfinite(predictions))
    assert model.n_rules >= 1
    assert model.rules_[0].krls.dictionary_size <= model.config.max_dictionary_size


def test_constructor_and_batch_validation_paths():
    config = VSEPLKRLSConfig()
    with pytest.raises(ValueError):
        VSEPLKRLS(config, alpha=0.2)
    with pytest.raises(TypeError):
        VSEPLKRLS(not_a_parameter=1)
    with pytest.raises(ValueError):
        configured_model().fit([], [])
    with pytest.raises(ValueError):
        configured_model().fit([[0.0], [1.0]], [0.0])


def test_unfitted_prediction_clipping_and_zero_activation_fallback():
    model = VSEPLKRLS(clip_inputs=True, initial_prediction=0.25)
    assert model.predict_one([2.0]) == pytest.approx(0.25)
    weights = model._normalized_activations(np.array([0.0, 0.0]))
    assert np.allclose(weights, [0.5, 0.5])


def test_long_regime_changing_stream_stays_finite_and_memory_bounded():
    rng = np.random.default_rng(20260816)
    n = 1500
    index = np.arange(n, dtype=float)
    x = np.column_stack(
        [
            (np.sin(index / 19.0) + 1.0) / 2.0,
            (np.cos(index / 37.0) + 1.0) / 2.0,
            index / (n - 1),
        ]
    )
    y = np.where(
        index < 500,
        0.3 * np.sin(index / 23.0),
        np.where(index < 1000, 0.2 * np.cos(index / 11.0), -0.15 + 0.25 * x[:, 0]),
    ) + rng.normal(0.0, 0.01, n)
    model = VSEPLKRLS(
        alpha=0.08,
        beta_initial=0.10,
        beta_min=0.001,
        beta_recovery_rate=0.25,
        error_threshold=0.8,
        error_normalization="running_std",
        max_rules=8,
        max_dictionary_size=10,
        replacement_strategy="least_used",
        forgetting_factor=0.995,
        dictionary_usage_decay=0.99,
        enable_rule_merging=True,
        adapt_kernel_width=False,
        kernel_sigma=0.2,
    )
    predictions = np.asarray([model.learn_one(row, target) for row, target in zip(x, y)])
    summary = model.summary()
    assert np.isfinite(predictions).all()
    assert model.n_seen_ == n
    assert 1 <= model.n_rules <= 8
    assert all(rule.krls.dictionary_size <= 10 for rule in model.rules_)
    assert model.config.beta_min <= model.beta_ <= model.config.beta_max
    assert 0 <= summary["dictionary_replacement_rate"] <= 1
