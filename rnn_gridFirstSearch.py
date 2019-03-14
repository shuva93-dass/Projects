# Recurrent Neural Network grid search code

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
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
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

def build_regressor(optimizer, loss):
	# Dropout layers are preserved in comments for ease
	# of use in case you want to see how they affect the
	# search
	regressor  =  Sequential()
	regressor.add(LSTM(units = 80, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
	#regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 80, return_sequences = True))
	#regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 80, return_sequences = True))
	#regressor.add(Dropout(0.2))
	regressor.add(LSTM(units = 80))
	#regressor.add(Dropout(0.2))
	regressor.add(Dense(units = 1))
	regressor.compile(optimizer = optimizer, loss = loss, metrics = [r2.coeff_determination])
	return  regressor
model_regressor  =  KerasRegressor(build_fn  =  build_regressor)
parameters = {'batch_size': [64],
                'epochs': [50],
                'optimizer': ['adam'],
				'loss': ['mean_squared_error', 'mean_absolute_error', 'logcosh']}
grid_search = GridSearchCV(estimator = model_regressor,
                              param_grid = parameters,
							  scoring = 'r2',
							  cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
# Warning: output of below commented print function is really
# hard to read; uncomment at own peril
#print('Results:', grid_search.cv_results_)
print('Best parameters:', grid_search.best_params_)
print('Best R^2 score:', grid_search.best_score_)
n = len(X_train)
k = len(features)
adj_r2 = 1 - (((1 - grid_search.best_score_) * (n - 1)) / (n - k - 1))
print('Best adjusted R^2 score:', adj_r2)
