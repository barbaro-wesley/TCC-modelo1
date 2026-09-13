from __future__ import annotations


import numpy as np

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


def _lgb_params():
    return dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )


def fit_lgbm(X: np.ndarray, y: np.ndarray, model=None):
    if lgb is None:
        raise RuntimeError("lightgbm nao instalado")
    if model is None:
        model = lgb.LGBMRegressor(**_lgb_params())
        model.fit(X, y)
    return model


def predict_lgbm(model, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def fit_xgb(X: np.ndarray, y: np.ndarray, model=None):
    if xgb is None:
        raise RuntimeError("xgboost nao instalado")
    if model is None:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(X, y)
    return model
