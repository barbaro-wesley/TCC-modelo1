from __future__ import annotations

from typing import Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def naive_predict(y_hist: np.ndarray, n: int = 1) -> np.ndarray:
    return np.repeat(y_hist[-1], n)


def ma_predict(y_hist: np.ndarray, window: int = 4, n: int = 1) -> np.ndarray:
    w = min(window, len(y_hist))
    return np.repeat(float(np.mean(y_hist[-w:])), n)


def _select_arima(y: np.ndarray, max_p: int = 1, max_d: int = 1, max_q: int = 1):
    best = None
    best_aic = np.inf
    y = np.asarray(y, dtype=float)
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == d == q == 0:
                    continue
                try:
                    m = ARIMA(y, order=(p, d, q), trend="n")
                    r = m.fit(method_kwargs={"warn_convergence": False})
                    if r.aic < best_aic:
                        best_aic = r.aic
                        best = r
                except Exception:
                    continue
    if best is None:
        r = ARIMA(y, order=(1, 1, 0)).fit()
        return r
    return best


def arima_forecast(y_hist: np.ndarray, steps: int = 1, model=None) -> Tuple[np.ndarray, object]:
    if model is None:
        model = _select_arima(y_hist)
        fc = np.asarray(model.forecast(steps=steps), dtype=float)
        return fc, model
    try:
        model = model.append([y_hist[-1]], refit=False)
        fc = np.asarray(model.forecast(steps=steps), dtype=float)
        return fc, model
    except Exception:
        model = _select_arima(y_hist)
        fc = np.asarray(model.forecast(steps=steps), dtype=float)
        return fc, model


def arimax_forecast(
    y_hist: np.ndarray,
    X_hist: np.ndarray,
    X_future: np.ndarray,
    steps: int = 1,
    model=None,
) -> Tuple[np.ndarray, object]:
    y_hist = np.asarray(y_hist, dtype=float)
    X_hist = np.asarray(X_hist, dtype=float)
    X_future = np.asarray(X_future, dtype=float).reshape(steps, -1)
    if model is None:
        best = None
        best_aic = np.inf
        for order in ((1, 1, 0), (1, 1, 1), (0, 1, 1)):
            try:
                m = SARIMAX(y_hist, exog=X_hist, order=order, trend="n")
                r = m.fit(disp=False, maxiter=40)
                if r.aic < best_aic:
                    best_aic = r.aic
                    best = r
            except Exception:
                continue
        model = best or SARIMAX(y_hist, exog=X_hist, order=(1, 1, 0)).fit(disp=False)
        fc = np.asarray(model.forecast(steps=steps, exog=X_future), dtype=float)
        return fc, model
    try:
        model = model.append([y_hist[-1]], exog=X_hist[-1:], refit=False)
        fc = np.asarray(model.forecast(steps=steps, exog=X_future), dtype=float)
        return fc, model
    except Exception:
        return arimax_forecast(y_hist, X_hist, X_future, steps=steps, model=None)
