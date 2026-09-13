import numpy as np
import pytest

from vs_epl_krls.kernels import rbf_features, rbf_kernel, rbf_kernel_matrix


def test_rbf_kernel_identity_symmetry_and_known_value():
    assert rbf_kernel([0.0, 1.0], [0.0, 1.0], sigma=0.5) == pytest.approx(1.0)
    expected = np.exp(-1.0 / (2.0 * 0.5**2))
    assert rbf_kernel([0.0], [1.0], sigma=0.5) == pytest.approx(expected)
    assert rbf_kernel([1.0], [0.0], sigma=0.5) == pytest.approx(expected)


def test_rbf_features_support_per_center_widths():
    values = rbf_features([0.0], [[0.0], [1.0]], [0.2, 1.0])
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(np.exp(-0.5))


def test_rbf_kernel_matrix_is_positive_semidefinite():
    x = np.linspace(0, 1, 8).reshape(-1, 1)
    gram = rbf_kernel_matrix(x, sigma=0.3)
    assert np.allclose(gram, gram.T)
    assert np.linalg.eigvalsh(gram).min() > -1e-10


@pytest.mark.parametrize(
    "call",
    [
        lambda: rbf_kernel([0.0], [0.0, 1.0]),
        lambda: rbf_kernel([0.0], [0.0], sigma=0.0),
        lambda: rbf_features([0.0], [[0.0]], [-1.0]),
        lambda: rbf_kernel_matrix([[0.0]], [[0.0, 1.0]]),
    ],
)
def test_kernel_validation(call):
    with pytest.raises(ValueError):
        call()
