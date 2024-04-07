#from src import random_walker as rw
from src import data_generator as dg
from numpy import array
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

#simulation parameters
n_steps = 1000
n_datasets = 1000
n_training_steps = 900

# training parameters
epochs  = 1000

dat = dg.Data(n_steps,n_training_steps,amplitude=1)
x_training_data,y_training_data = dat.generate_datasets(n_datasets=n_datasets)


X = array(x_training_data)
y = array(y_training_data)

# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1)) 

# define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_training_steps, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(101))

# compile
model.compile(optimizer='adam', loss='mse')



# train model
model.fit(X, y, epochs=epochs, verbose=1)


testdat = dg.Data(n_steps,
                  n_training_steps)
x_test_data,y_test_data = testdat.generate_datasets(n_datasets=1,seed=n_datasets+5)
x_test_data = array(x_test_data)
x_test_data = x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1],1)
yhat = model.predict(x_test_data, verbose=0)
plt.plot(x_test_data[0]+y_test_data[0])
plt.plot(x_test_data[0]+list(yhat[0]))