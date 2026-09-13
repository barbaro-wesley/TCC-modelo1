"""Kernel functions used by the VS-ePL-KRLS consequents."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_vector(x: ArrayLike, *, name: str) -> NDArray[np.float64]:
    value = np.asarray(x, dtype=float)
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional numeric vector")
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite values")
    return value


def rbf_kernel(x: ArrayLike, y: ArrayLike, sigma: float = 0.5) -> float:
    """Return ``exp(-||x-y||² / (2 sigma²))``.

    This convention is equivalent to the Gaussian expression in Eq. 9 of the
    VS-ePL-KRLS article when ``sigma`` denotes the kernel width ``nu``.
    """

    x_array = _as_vector(x, name="x")
    y_array = _as_vector(y, name="y")
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must have the same shape")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive number")
    distance_squared = float(np.dot(x_array - y_array, x_array - y_array))
    return float(np.exp(-distance_squared / (2.0 * sigma**2)))


def rbf_features(
    x: ArrayLike,
    dictionary: ArrayLike,
    widths: ArrayLike | float,
    *,
    min_width: float = 1e-8,
) -> NDArray[np.float64]:
    """Evaluate an RBF centred at every dictionary element.

    A scalar width is broadcast.  A vector permits the per-element widths used
    in the source algorithm.
    """

    x_array = _as_vector(x, name="x")
    centers = np.asarray(dictionary, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != x_array.size:
        raise ValueError("dictionary must be a 2-D array with x.shape[0] columns")
    if not np.all(np.isfinite(centers)):
        raise ValueError("dictionary must contain finite values")
    width_array = np.asarray(widths, dtype=float)
    if width_array.ndim == 0:
        width_array = np.full(centers.shape[0], float(width_array))
    if width_array.shape != (centers.shape[0],):
        raise ValueError("widths must be scalar or have one value per center")
    if not np.all(np.isfinite(width_array)) or np.any(width_array <= 0):
        raise ValueError("widths must contain finite positive values")
    safe_widths = np.maximum(width_array, float(min_width))
    distances_squared = np.sum((centers - x_array[None, :]) ** 2, axis=1)
    return np.exp(-distances_squared / (2.0 * safe_widths**2))


def rbf_kernel_matrix(
    x: ArrayLike,
    y: ArrayLike | None = None,
    sigma: float = 0.5,
) -> NDArray[np.float64]:
    """Return the pairwise RBF Gram matrix for two 2-D arrays."""

    x_array = np.asarray(x, dtype=float)
    y_array = x_array if y is None else np.asarray(y, dtype=float)
    if x_array.ndim != 2 or y_array.ndim != 2:
        raise ValueError("x and y must be two-dimensional arrays")
    if x_array.shape[1] != y_array.shape[1]:
        raise ValueError("x and y must have the same number of features")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("x and y must contain finite values")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a finite positive number")
    distance_squared = np.sum(
        (x_array[:, None, :] - y_array[None, :, :]) ** 2,
        axis=2,
    )
    return np.exp(-distance_squared / (2.0 * sigma**2))
