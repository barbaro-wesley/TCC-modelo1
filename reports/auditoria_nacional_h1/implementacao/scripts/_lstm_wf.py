from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.lstm import fit_lstm, predict_lstm  # noqa: E402


def main():
    npz_path = Path(sys.argv[1])
    n_min = int(sys.argv[2])
    horizon = int(sys.argv[3])
    refit_every = int(sys.argv[4]) if len(sys.argv) > 4 else 26
    seq_len = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    data = np.load(npz_path)
    X, y = data["X"], data["y"]
    yhat = np.full(len(y), np.nan)
    model = None
    last = -10**9
    for t in range(max(n_min, seq_len), len(y)):
        te = max(0, t - horizon + 1)
        if te < seq_len + 8:
            continue
        if t - last >= refit_every or model is None:
            model = fit_lstm(X[:te], y[:te], seq_len=seq_len, epochs=8, hidden=8)
            last = t
        yhat[t] = predict_lstm(model, X[:te], seq_len=seq_len)
    out = npz_path.with_name(npz_path.stem + "_yhat.npy")
    np.save(out, yhat)
    print("lstm_ok", int(np.isfinite(yhat).sum()))


if __name__ == "__main__":
    main()
