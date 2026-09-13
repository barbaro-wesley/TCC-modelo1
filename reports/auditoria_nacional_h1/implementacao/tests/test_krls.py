import numpy as np
import pytest

from vs_epl_krls.krls import SparseKRLS


def test_krls_initialization_and_prediction_after_update():
    model = SparseKRLS(sigma=0.2, regularization=1e-4)
    result = model.update([0.25], 0.8)
    assert result.action == "initialized"
    assert model.dictionary_size == 1
    assert model.predict([0.25]) == pytest.approx(0.8, rel=2e-4)


def test_krls_inserts_a_novel_dictionary_element():
    model = SparseKRLS(sigma=0.1, novelty_factor=0.1, max_dictionary_size=4)
    model.update([0.0], 0.0)
    result = model.update([1.0], 1.0)
    assert result.action == "inserted"
    assert model.dictionary_size == 2
    assert model.Q_.shape == (2, 2)
    assert np.all(np.isfinite(model.coefficients_))


def test_krls_recursively_updates_a_coherent_sample():
    model = SparseKRLS(sigma=0.5, novelty_factor=1.0)
    model.update([0.5], 0.0)
    before = model.predict([0.5])
    result = model.update([0.5], 1.0)
    after = model.predict([0.5])
    assert result.action == "updated"
    assert abs(1.0 - after) < abs(1.0 - before)


@pytest.mark.parametrize("strategy", ["oldest", "least_used"])
def test_krls_enforces_dictionary_limit_with_explicit_replacement(strategy):
    model = SparseKRLS(
        sigma=0.05,
        novelty_factor=0.01,
        max_dictionary_size=2,
        replacement_strategy=strategy,
    )
    actions = [model.update([value], value).action for value in (0.0, 0.5, 1.0)]
    assert actions == ["initialized", "inserted", "replaced"]
    assert model.dictionary_size == 2
    assert model.n_replacements_ == 1


def test_krls_full_dictionary_can_disable_replacement():
    model = SparseKRLS(
        sigma=0.05,
        novelty_factor=0.01,
        max_dictionary_size=1,
        replacement_strategy="none",
    )
    model.update([0.0], 0.0)
    result = model.update([1.0], 1.0)
    assert result.action == "updated"
    assert model.dictionary_size == 1


def test_recent_usage_decay_drives_least_used_replacement_statistics():
    model = SparseKRLS(
        sigma=0.05,
        novelty_factor=0.01,
        max_dictionary_size=2,
        replacement_strategy="least_used",
        usage_decay=0.5,
    )
    model.update([0.0], 0.0)
    model.update([1.0], 1.0)
    model.update([0.0], 0.0)
    assert model.usage_[0] > model.usage_[1]
    inspected = model.inspect()
    assert inspected["usage"] == pytest.approx(model.usage_.tolist())
    assert inspected["replacement_rate"] == 0.0


def test_krls_rejects_invalid_usage_decay():
    with pytest.raises(ValueError, match="usage_decay"):
        SparseKRLS(usage_decay=0.0)


def test_krls_repeated_values_and_width_adaptation_stay_finite():
    model = SparseKRLS(sigma=0.2, regularization=1e-8, novelty_factor=1.0)
    for _ in range(20):
        model.update([0.2, 0.2], 0.7)
    changed = model.adapt_widths([0.5, 0.5], error=0.1, activation=1.0)
    assert isinstance(changed, bool)
    assert np.all(np.isfinite(model.Q_))
    assert np.isfinite(model.predict([0.2, 0.2]))


def test_krls_rejects_non_finite_values_and_bad_dimensions():
    model = SparseKRLS()
    with pytest.raises(ValueError):
        model.update([np.nan], 1.0)
    model.update([0.0, 0.0], 0.0)
    with pytest.raises(ValueError):
        model.predict([0.0])
