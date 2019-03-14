# RNN Projects

RNN_Model.py:
This file is where we used the RNN model trained on the historic google stock prices to predict the future ones. We plotted the graph to show how closely accurate our prediction  was to the actual stock prices.

r2.py:
This file defines the metric we used to assess our RNN model which is adjusted R square.

rnn_gridFirstSearch.py:
We also used Grid First Search on RNN where we experimented with different types of loss functions namely, 'mean_squared_error', 'mean_absolute_error', 'logcosh' to see which loss function works best with RNN model also considering multiple features to further explore the effects of number of features on accuracy which we measured by adjusted R2.
