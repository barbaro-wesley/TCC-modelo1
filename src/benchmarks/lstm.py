from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from benchmarks.sequences import make_sequences


class TinyLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_lstm(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 12,
    epochs: int = 25,
    hidden: int = 16,
    lr: float = 1e-2,
    model=None,
    seed: int = 0,
    dates=None,
):
    torch.manual_seed(seed)
    Xs, ys = make_sequences(X, y, seq_len, dates=dates)
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
    x = torch.from_numpy(X[-seq_len:].astype(np.float32)[None, ...])
    model.eval()
    with torch.no_grad():
        return float(model(x).cpu().numpy().ravel()[0])
