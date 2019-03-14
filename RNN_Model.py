# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Import libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
# CuDNNLSTM is imported because it runs much faster
# on GPUs. If running on a computer without a CUDA-
# enabled GPU, delete "CuDNNLSTM as" in the line below.
from keras.layers import CuDNNLSTM as LSTM
# Dropout is imported because we did some experiments
# that included dropping out some neurons
# Based on the results of these experiments, we believe
# dropout is not helpful for this model
from keras.layers import Dropout
import r2

# Importing the training set
dataset_train = pd.read_csv('GSPC_train.csv')
features = ['Open', 'High', 'Close']
training_set = dataset_train.loc[:, features].values

# Feature Scaling
sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 time stamps and 1 output
X_train=[]
Y_train=[]
for i in range(60, training_set_scaled.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0:len(features)])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshaping
if len(features) == 1:
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Dropout layers are preserved in comments for ease
# of use in case you want to see how they affect the
# model

# Adding the first LSTM layer
regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
#regressor.add(Dropout(0.1))

# Adding the 2nd LSTM layer
regressor.add(LSTM(units = 80, return_sequences = True))
#regressor.add(Dropout(0.1))

# Adding the 3rd LSTM layer
regressor.add(LSTM(units = 80, return_sequences = True))
#regressor.add(Dropout(0.1))

# Adding the 4th LSTM layer
regressor.add(LSTM(units = 80))
#regressor.add(Dropout(0.1))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN

# Keras defines 5 loss functions for regression
# Experiments on them resulted in the following
# performance levels:
#	Mean squared error: good
#	Mean absolute error: good
#	Mean absolute percentage error: bad
#	Mean squared logarithmic error: mediocre
#	Logcosh: good
# Further experiments suggest that mean squared
# error with 3 input features provides best
# performance
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [r2.adjusted_r2])

# Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, batch_size = 64, epochs = 50)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of October 2018
dataset_test = pd.read_csv('GSPC_test.csv')
real_stock_price = dataset_test.loc[:, ['Open']].values

# Turning it into the test dataset
dataset_total = pd.concat((dataset_train[features], dataset_test[features]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = sc.transform(inputs)
X_test = []
Y_test = []
for i in range(60, 60 + len(dataset_test)):
	X_test.append(inputs[i-60:i, 0:len(features)])
	Y_test.append(inputs[i, 0])
X_test = np.array(X_test)
Y_test = np.array(Y_test)
if len(features) == 1:
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	
# Evaluating the model
metrics = regressor.evaluate(x = X_test, y = Y_test, batch_size = 64)
print(regressor.metrics_names)
print(metrics)

# Getting the predicted stock price
predicted_stock_price = regressor.predict(X_test)
# Reshaping the predicted stock price for inverse transform
if len(features) != 1:
	predicted_stock_price = np.append(predicted_stock_price, np.zeros((len(dataset_test), len(features) - 1)), axis = 1)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Reshaping it for showing it on the plot
if len(features) != 1:
	dims = list(range(1, len(features))
	predicted_stock_price = np.delete(predicted_stock_price, dims, axis = 1)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
