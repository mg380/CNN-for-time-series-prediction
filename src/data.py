"""Build supervised forecasting datasets from synthetic random-walk series.

Each example is a sliding window: the first ``window`` steps of a series are the
input ``X``, the remaining steps are the target ``Y`` (a direct multi-step
forecast). Train / validation / test splits use disjoint seed ranges so no
series is shared across splits.
"""
import numpy as np

from src.random_walker import RandomWalk


def make_series(seed, n_steps, amplitude=1.0, period=10, **kw):
    """Return one series as a 1-D array of length ``n_steps + 1``."""
    rw = RandomWalk(seed=seed, amplitude=amplitude, period=period, **kw)
    rw.generate(n_steps)
    return np.asarray(rw.chain, dtype=np.float32)


def build_dataset(n_series, n_steps, window, amplitude=1.0, period=10,
                  seed_start=0, **kw):
    """Return (X, Y) with shapes (n_series, window, 1) and (n_series, horizon)."""
    X, Y = [], []
    for i in range(n_series):
        chain = make_series(seed_start + i, n_steps, amplitude, period, **kw)
        X.append(chain[:window])
        Y.append(chain[window:])
    X = np.asarray(X, dtype=np.float32)[..., None]
    Y = np.asarray(Y, dtype=np.float32)
    return X, Y


def build_splits(n_train, n_val, n_test, n_steps, window,
                 amplitude=1.0, period=10, **kw):
    """Return ((Xtr, Ytr), (Xva, Yva), (Xte, Yte)) over disjoint series."""
    train = build_dataset(n_train, n_steps, window, amplitude, period,
                          seed_start=0, **kw)
    val = build_dataset(n_val, n_steps, window, amplitude, period,
                        seed_start=n_train, **kw)
    test = build_dataset(n_test, n_steps, window, amplitude, period,
                         seed_start=n_train + n_val, **kw)
    return train, val, test
