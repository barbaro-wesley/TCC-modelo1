import numpy as np
import pytest

from vs_epl_krls import EPLKRLSFixedBeta, VSEPLKRLS, VSEPLKRLSConfig


def test_large_error_increases_beta_and_small_error_decreases_it():
    model = VSEPLKRLS(
        beta_initial=0.2,
        alpha_vs1=0.8,
        alpha_vs2=0.5,
        error_threshold=0.1,
        adapt_kernel_width=False,
        enable_rule_merging=False,
    )
    assert model._adapt_beta(0.2) == pytest.approx(0.25)
    model.beta_ = 0.2
    assert model._adapt_beta(0.05) == pytest.approx(0.1)


def test_beta_is_clipped_to_configured_bounds():
    model = VSEPLKRLS(
        beta_initial=0.2,
        beta_min=0.15,
        beta_max=0.22,
        alpha_vs1=0.5,
        alpha_vs2=0.5,
        error_threshold=0.1,
    )
    assert model._adapt_beta(1.0) == pytest.approx(0.22)
    model.beta_ = 0.2
    assert model._adapt_beta(0.0) == pytest.approx(0.15)


def test_optional_beta_recovery_escapes_a_collapsed_floor_after_large_error():
    paper = VSEPLKRLS(
        beta_initial=0.2,
        beta_min=0.001,
        alpha_vs1=0.9,
        error_threshold=1.0,
        beta_recovery_rate=0.0,
    )
    recovered = VSEPLKRLS(
        beta_initial=0.2,
        beta_min=0.001,
        alpha_vs1=0.9,
        error_threshold=1.0,
        beta_recovery_rate=0.5,
    )
    paper.beta_ = recovered.beta_ = 0.001
    assert paper._adapt_beta(3.0) == pytest.approx(0.001 / 0.9)
    assert recovered._adapt_beta(3.0) == pytest.approx(0.1005)


def test_arousal_and_thresholds_use_previous_beta():
    model = VSEPLKRLS(
        beta_initial=0.2,
        alpha_vs1=0.5,
        alpha_vs2=0.5,
        error_threshold=0.01,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )
    model.learn_one([0.0], 0.0)
    model.learn_one([1.0], 1.0)
    event = model.get_history()[-1]
    assert event["beta_previous"] == pytest.approx(0.2)
    assert event["tau"] == pytest.approx(0.2)
    assert event["gamma"] == pytest.approx(0.8)
    assert event["beta"] == pytest.approx(0.4)


def test_fixed_beta_ablation_never_changes_beta():
    model = EPLKRLSFixedBeta(
        beta_initial=0.2,
        error_threshold=0.01,
        enable_rule_merging=False,
        adapt_kernel_width=False,
    )
    for x, y in [([0.0], 0.0), ([1.0], 1.0), ([0.5], 0.0)]:
        model.learn_one(x, y)
    assert {event["beta"] for event in model.get_history()} == {0.2}


def test_running_range_normalization_uses_only_past_targets():
    model = VSEPLKRLS(error_normalization="running_range")
    model._target_stats.update(2.0)
    model._target_stats.update(4.0)
    assert model._normalized_error(1.0) == pytest.approx(0.5)


def test_fixed_and_running_std_error_normalization():
    fixed = VSEPLKRLS(error_normalization="fixed", error_scale=4.0)
    assert fixed._normalized_error(2.0) == pytest.approx(0.5)
    running = VSEPLKRLS(error_normalization="running_std")
    running._target_stats.update(1.0)
    running._target_stats.update(3.0)
    assert running._normalized_error(np.sqrt(2.0)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "parameters",
    [
        {"alpha": 0.0},
        {"beta_initial": 0.2, "beta_min": 0.3},
        {"merge_threshold": 2.0},
        {"regularization": 0.0},
        {"novelty_factor": -1.0},
        {"max_rules": 0},
        {"forgetting_factor": 0.0},
        {"dictionary_usage_decay": 0.0},
        {"beta_recovery_rate": 1.1},
        {"max_width_relative_change": 2.0},
        {"input_bounds": (1.0, 0.0)},
        {"initial_prediction": np.inf},
    ],
)
def test_configuration_rejects_invalid_values(parameters):
    with pytest.raises(ValueError):
        VSEPLKRLSConfig(**parameters)


def test_fixed_beta_accepts_config_object():
    model = EPLKRLSFixedBeta(VSEPLKRLSConfig(beta_initial=0.2))
    assert model.config.variable_beta is False
    assert model.beta_ == pytest.approx(0.2)
