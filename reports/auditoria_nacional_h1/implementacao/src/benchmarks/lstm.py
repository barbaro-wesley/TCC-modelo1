from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # optional legacy benchmark
    torch = None
    nn = None


if nn is not None:
    class TinyLSTM(nn.Module):
        def __init__(self, n_feat: int, hidden: int = 16):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)
else:
    class TinyLSTM:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch nao instalado; instale requirements-benchmarks.txt")


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for t in range(seq_len, len(y)):
        xs.append(X[t - seq_len : t])
        ys.append(y[t])
    if not xs:
        return np.zeros((0, seq_len, X.shape[1])), np.zeros((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def fit_lstm(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 12,
    epochs: int = 25,
    hidden: int = 16,
    lr: float = 1e-2,
    model=None,
    seed: int = 0,
):
    if torch is None or nn is None:
        raise RuntimeError("torch nao instalado; instale requirements-benchmarks.txt")
    torch.manual_seed(seed)
    Xs, ys = make_sequences(X, y, seq_len)
    if len(ys) < 8:
        return None
    if model is None:
        model = TinyLSTM(X.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(ys)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    return model


def predict_lstm(model, X: np.ndarray, seq_len: int = 12) -> float:
    if model is None or len(X) < seq_len:
        return float("nan")
    if torch is None:
        raise RuntimeError("torch nao instalado; instale requirements-benchmarks.txt")
    x = torch.from_numpy(X[-seq_len:].astype(np.float32)[None, ...])
    model.eval()
    with torch.no_grad():
        return float(model(x).cpu().numpy().ravel()[0])
