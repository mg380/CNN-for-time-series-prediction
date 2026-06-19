#from src import random_walker as rw
from src import data_generator as dg
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D

#simulation parameters
n_steps = 1000
n_datasets = 1000
n_training_steps = 900
amplitude = 1

# training parameters
epochs  = 200

dat = dg.Data(n_steps,n_training_steps,amplitude=amplitude)
x_training_data,y_training_data = dat.generate_datasets(n_datasets=n_datasets)


X = array(x_training_data)
y = array(y_training_data)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))

# number of future steps to predict (chain has n_steps+1 points, minus the
# training window)
n_output = y.shape[1]

# define model
model = Sequential()
model.add(Input(shape=(n_training_steps, 1)))
model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_output))

# compile
model.compile(optimizer='adam', loss='mse')
model.summary()


# train model
model.fit(X, y, epochs=epochs, verbose=1)


testdat = dg.Data(n_steps,
                  n_training_steps,
                  amplitude=amplitude)
x_test_data,y_test_data = testdat.generate_datasets(n_datasets=1,seed=n_datasets+5)
x_test_data = array(x_test_data)
x_test_data = x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1],1)
yhat = model.predict(x_test_data, verbose=0)

# build the full series: the known input window followed by either the true
# continuation or the model's prediction
history = x_test_data[0].flatten()
true_future = array(y_test_data[0]).flatten()
pred_future = array(yhat[0]).flatten()

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
plt.plot(actual_series, label='actual', linewidth=1)
plt.plot(forecast_x, forecast_y, label='predicted', linewidth=1)
plt.axvline(split, color='grey', linestyle='--', label='forecast start')
plt.title('CNN time-series forecast: last %d steps' % n_output)
plt.xlabel('time step')
plt.ylabel('value')
plt.legend()
plt.tight_layout()
plt.savefig('forecast.png', dpi=120)
print('Saved plot to forecast.png')
plt.show()
