"""Forecasting models for the comparison study.

Two families share a common ``predict`` interface so the evaluation harness can
treat them uniformly:

* **Baselines** (no learning): persistence, seasonal-naive, ridge regression.
* **Neural networks** (Keras): MLP, CNN, LSTM, GRU, TCN -- each ends in a
  Dropout layer so the same MC-Dropout uncertainty estimator applies to all.
"""
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Flatten, Dropout, Conv1D,
                                      MaxPooling1D, LSTM, GRU,
                                      GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class Persistence:
    """Predict the last observed value, repeated across the horizon."""
    name = "persistence"

    def fit(self, X, Y):
        return self

    def predict(self, X, horizon):
        last = X[:, -1, 0]
        return np.repeat(last[:, None], horizon, axis=1)


class SeasonalNaive:
    """Repeat the most recent seasonal cycle. A strong baseline for periodic data."""

    def __init__(self, season):
        self.season = int(season)
        self.name = "seasonal_naive"

    def fit(self, X, Y):
        return self

    def predict(self, X, horizon):
        last_season = X[:, -self.season:, 0]          # (n, season)
        idx = np.arange(horizon) % self.season
        return last_season[:, idx]


class LinearRidge:
    """Multi-output ridge regression on the flattened input window."""

    def __init__(self, alpha=10.0):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)
        self.name = "linear"

    def fit(self, X, Y):
        self.model.fit(X[:, :, 0], Y)
        return self

    def predict(self, X, horizon):
        return self.model.predict(X[:, :, 0])


# --------------------------------------------------------------------------- #
# Neural builders (each returns an uncompiled Keras model)
# --------------------------------------------------------------------------- #
def build_mlp(window, horizon, dropout=0.2):
    return Sequential([
        Input(shape=(window, 1)),
        Flatten(),
        Dense(100, activation="relu"),
        Dropout(dropout),
        Dense(50, activation="relu"),
        Dropout(dropout),
        Dense(horizon),
    ], name="mlp")


def build_cnn(window, horizon, dropout=0.2):
    return Sequential([
        Input(shape=(window, 1)),
        Conv1D(64, 2, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(50, activation="relu"),
        Dropout(dropout),
        Dense(horizon),
    ], name="cnn")


def build_lstm(window, horizon, dropout=0.2):
    return Sequential([
        Input(shape=(window, 1)),
        LSTM(64),
        Dropout(dropout),
        Dense(50, activation="relu"),
        Dense(horizon),
    ], name="lstm")


def build_gru(window, horizon, dropout=0.2):
    return Sequential([
        Input(shape=(window, 1)),
        GRU(64),
        Dropout(dropout),
        Dense(50, activation="relu"),
        Dense(horizon),
    ], name="gru")


def build_tcn(window, horizon, dropout=0.2):
    """Temporal Convolutional Network: stacked dilated *causal* convolutions.

    Dilations grow the receptive field exponentially without pooling, so the
    last timestep sees far into the past.
    """
    inp = Input(shape=(window, 1))
    x = inp
    for dilation in (1, 2, 4, 8, 16):
        x = Conv1D(32, 2, padding="causal", dilation_rate=dilation,
                   activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(dropout)(x)
    out = Dense(horizon)(x)
    return Model(inp, out, name="tcn")


NEURAL_BUILDERS = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "lstm": build_lstm,
    "gru": build_gru,
    "tcn": build_tcn,
}


def train_keras(model, train, val, epochs=80, patience=8, batch_size=32,
                verbose=0):
    """Compile and fit a Keras model with early stopping on validation loss."""
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True)
    history = model.fit(train[0], train[1], validation_data=val,
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[es], verbose=verbose)
    return history
