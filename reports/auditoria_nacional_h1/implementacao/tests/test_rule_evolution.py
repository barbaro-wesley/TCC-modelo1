import numpy as np
import pytest

from vs_epl_krls import EvolvingRule, SparseKRLS, VSEPLKRLS


def make_rule(center=(0.2, 0.4)):
    consequent = SparseKRLS()
    consequent.update(center, 0.5)
    return EvolvingRule(
        rule_id=1,
        center=np.asarray(center, dtype=float),
        dispersion=np.full(len(center), 0.05),
        arousal=0.0,
        krls=consequent,
        created_at=0,
        last_updated=0,
    )


def test_compatibility_and_arousal_match_published_equations():
    rule = make_rule()
    expected = 1.0 - np.linalg.norm(np.array([0.4, 0.4]) - rule.center) / 2.0
    assert rule.compatibility([0.4, 0.4]) == pytest.approx(expected)
    arousal = rule.update_arousal(expected, beta=0.2)
    assert arousal == pytest.approx(0.2 * (1.0 - expected))


def test_literal_paper_center_update():
    rule = make_rule()
    rule.arousal = 0.25
    old = rule.center.copy()
    sample = np.array([0.8, 0.6])
    expected = old + 0.1 * old ** (1.0 - rule.arousal) * (sample - old)
    rule.update_antecedent(sample, alpha=0.1, compatibility=0.9, step=1, mode="paper")
    assert np.allclose(rule.center, expected)
    assert np.all(rule.dispersion > 0)


def test_new_rule_decision_uses_minimum_arousal():
    model = VSEPLKRLS(
        variable_beta=False,
        arousal_threshold=0.16,
        enable_rule_merging=False,
        adapt_kernel_width=False,
        input_bounds=(0.0, 1.0),
    )
    model.learn_one([0.0], 0.0)
    model.learn_one([1.0], 1.0)
    assert model.n_rules == 2
    # Close to rule 1: at least one rule remains below tau, so no third rule.
    model.learn_one([0.02], 0.02)
    assert model.n_rules == 2


def test_rule_merging_records_event_and_preserves_consequent():
    model = VSEPLKRLS(
        variable_beta=False,
        arousal_threshold=0.01,
        merge_threshold=0.97,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )
    model.learn_one([0.1], 0.1)
    model.learn_one([0.3], 0.3)
    assert model.n_rules == 2
    model.config = type(model.config)(
        **{**model.summary()["config"], "enable_rule_merging": True, "merge_threshold": 0.5}
    )
    dropped = model._merge_redundant(0.5)
    assert len(dropped) == 1
    assert model.n_rules == 1
    assert model.n_rule_merges_ == 1
    assert model.rules_[0].krls.dictionary_size >= 1


def test_compatibility_threshold_is_an_explicit_optional_trigger():
    model = VSEPLKRLS(
        variable_beta=False,
        arousal_threshold=1.0,
        compatibility_threshold=0.9,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )
    model.learn_one([0.0], 0.0)
    model.learn_one([1.0], 1.0)
    assert model.n_rules == 2
