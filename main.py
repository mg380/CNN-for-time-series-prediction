"""Quick single-model demo: train the CNN and plot one forecast.

For the full multi-model evaluation study see ``run_study.py``.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.data import build_dataset
from src.models import build_cnn, train_keras

# simulation parameters
n_steps = 1000
n_train = 1000
window = 900
amplitude = 1.0

# training parameters
epochs = 200

# training data: many series, each split into (window -> future)
X, y = build_dataset(n_train, n_steps, window, amplitude=amplitude, seed_start=0)
horizon = y.shape[1]

# a small validation split so we can early-stop and watch for overfitting
val_cut = int(0.9 * len(X))
train = (X[:val_cut], y[:val_cut])
val = (X[val_cut:], y[val_cut:])

model = build_cnn(window, horizon)
model.summary()
train_keras(model, train, val, epochs=epochs, patience=15, verbose=1)

# held-out test series (a seed disjoint from the training range)
x_test, y_test = build_dataset(1, n_steps, window, amplitude=amplitude,
                               seed_start=n_train + 5)
yhat = model.predict(x_test, verbose=0)

history = x_test[0].flatten()
true_future = y_test[0]
pred_future = yhat[0]

# the actual series is one continuous random walk, so plot it whole
actual_series = np.concatenate([history, true_future])
split = len(history)  # x-index where the forecast begins

# Anchor the forecast to the last observed value so it reads as a continuation.
# The model predicts all future steps jointly with no constraint to start from
# history[-1]; without this anchor the predicted line appears to "jump" at the
# boundary even though the underlying data is continuous there.
forecast_x = np.arange(split - 1, split + len(pred_future))
forecast_y = np.concatenate([[history[-1]], pred_future])

plt.figure(figsize=(12, 5))
plt.plot(actual_series, label="actual", linewidth=1)
plt.plot(forecast_x, forecast_y, label="predicted", linewidth=1)
plt.axvline(split, color="grey", linestyle="--", label="forecast start")
plt.title("CNN time-series forecast: last %d steps" % horizon)
plt.xlabel("time step")
plt.ylabel("value")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png", dpi=120)
print("Saved plot to forecast.png")
plt.show()
