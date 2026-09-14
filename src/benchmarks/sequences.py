"""Calendar-aware LSTM sequence construction, independent of the torch runtime."""

import numpy as np

from eval.temporal import weekly_history_starts


def make_sequences(X, y, seq_len, dates=None):
    """Training pair for origin t: X[t-seq_len:t] -> shifted target y[t]."""
    if seq_len < 1 or len(X) != len(y):
        raise ValueError("Expected a positive sequence length and aligned X/y")
    xs, ys = [], []
    starts = weekly_history_starts(len(y), dates)
    for t in range(seq_len, len(y)):
        if t - starts[t] < seq_len:
            continue
        xs.append(X[t - seq_len:t])
        ys.append(y[t])
    if not xs:
        return np.zeros((0, seq_len, X.shape[1])), np.zeros((0,))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
