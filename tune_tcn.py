"""Tune the TCN to try to beat the recurrent models.

The baseline TCN underfit. Two suspected causes:
  1. Receptive field ~32 steps < one full cycle (season=40) -> it can't see a
     whole oscillation.
  2. GlobalAveragePooling collapses the time axis, discarding phase information.

This script searches a few configurations on the *same* seeded train/val/test
split used by run_study.py, so results are directly comparable to the GRU
(RMSE 4.08) and LSTM (RMSE 4.54) references.

    STUDY_QUICK=1 uv run tune_tcn.py   # fast smoke test
    uv run tune_tcn.py                 # full search
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dropout, Dense, Add, Lambda,
                                      GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping

from src.data import build_splits
from src.metrics import rmse

SEED = 42
tf.keras.utils.set_random_seed(SEED)

QUICK = os.environ.get("STUDY_QUICK") == "1"
if QUICK:
    N_STEPS, WINDOW = 120, 100
    N_TRAIN, N_VAL, N_TEST = 24, 8, 16
    EPOCHS, PATIENCE = 3, 2
else:
    N_STEPS, WINDOW = 400, 300
    N_TRAIN, N_VAL, N_TEST = 300, 75, 150
    EPOCHS, PATIENCE = 100, 10
PERIOD = 10

# GRU / LSTM references on the identical split (from run_study.py)
REFERENCES = {"gru": 4.083, "lstm": 4.542, "old tcn (GAP, RF~32)": 5.374}


def build_tcn(window, horizon, filters=64, dilations=(1, 2, 4, 8, 16, 32, 64),
              kernel=2, dropout=0.1, readout="last", stacks=1, residual=True):
    """Flexible residual TCN.

    Receptive field per stack = 1 + 2*(kernel-1)*sum(dilations) (two convs/block).
    ``readout='last'`` takes the final causal timestep (full receptive field);
    ``'gap'`` global-average-pools (the old, lossy behaviour).
    """
    inp = Input(shape=(window, 1))
    x = inp
    for _ in range(stacks):
        for d in dilations:
            prev = x
            h = Conv1D(filters, kernel, padding="causal", dilation_rate=d,
                       activation="relu")(x)
            h = Dropout(dropout)(h)
            h = Conv1D(filters, kernel, padding="causal", dilation_rate=d,
                       activation="relu")(h)
            h = Dropout(dropout)(h)
            if residual:
                if prev.shape[-1] != filters:
                    prev = Conv1D(filters, 1, padding="same")(prev)
                x = Add()([prev, h])
            else:
                x = h
    if readout == "last":
        x = Lambda(lambda t: t[:, -1, :])(x)
    else:
        x = GlobalAveragePooling1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(horizon)(x)
    return Model(inp, out, name="tcn")


# Configurations to try (name -> kwargs)
CONFIGS = {
    "fix: last-step, RF~255":      dict(filters=64, dilations=(1, 2, 4, 8, 16, 32, 64), readout="last"),
    "ablation: same but GAP":      dict(filters=64, dilations=(1, 2, 4, 8, 16, 32, 64), readout="gap"),
    "wider (96 filters)":          dict(filters=96, dilations=(1, 2, 4, 8, 16, 32, 64), readout="last"),
    "deeper (2 stacks)":           dict(filters=64, dilations=(1, 2, 4, 8, 16, 32), readout="last", stacks=2),
    "more dropout (0.2)":          dict(filters=64, dilations=(1, 2, 4, 8, 16, 32, 64), readout="last", dropout=0.2),
}


def main():
    train, val, test = build_splits(N_TRAIN, N_VAL, N_TEST, N_STEPS, WINDOW,
                                    amplitude=1.0, period=PERIOD)
    Xtr, Ytr = train
    Xva, Yva = val
    Xte, Yte = test
    horizon = Ytr.shape[1]
    print(f"data: train{Xtr.shape} test{Xte.shape} horizon={horizon}\n")

    results = []
    for name, kw in CONFIGS.items():
        model = build_tcn(WINDOW, horizon, **kw)
        model.compile(optimizer="adam", loss="mse")
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                           restore_best_weights=True)
        hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva), epochs=EPOCHS,
                         batch_size=32, callbacks=[es], verbose=0)
        score = rmse(Yte, model.predict(Xte, verbose=0))
        results.append((name, score, model.count_params(), len(hist.history["loss"])))
        print(f"{name:30s} rmse={score:6.3f}  params={model.count_params():>7d}  "
              f"epochs={len(hist.history['loss'])}")

    print("\n=== ranked (with references) ===")
    table = results + [(f"[ref] {k}", v, 0, 0) for k, v in REFERENCES.items()]
    for name, score, params, _ in sorted(table, key=lambda r: r[1]):
        tag = "" if name.startswith("[ref]") else "  <-- tuned TCN"
        print(f"{name:34s} rmse={score:6.3f}{tag}")

    best = min(results, key=lambda r: r[1])
    print(f"\nbest TCN config: {best[0]}  (rmse={best[1]:.3f})")


if __name__ == "__main__":
    main()
