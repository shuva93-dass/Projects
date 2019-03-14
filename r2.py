# R^2 and adjusted R^2 regression functions
# Adjusted R^2 code written by Reed Mauzy
# R^2 code provided by the Stack Overflow community

from keras import backend as K

"""
The below explanations of what R^2 and adjusted R^2
are adapted from those provided in the post linked below:
https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4
Coefficient of determination (R^2)
Mathematically, R^2 is given by:
       Mean squared error
R^2 = --------------------
      Variance in Y values
It has some problems, but it does a fair job of
showing how well the predictions fit the ground truth.
It doesn't account for number of samples or number of
features, though, which is why adjusted R^2 exists.
The below code was provided in the post linked below:
https://stackoverflow.com/a/46969576
"""

def coeff_determination(y_true, y_pred):
	mse =  K.sum(K.square(y_true - y_pred))
	var = K.sum(K.square(y_true - K.mean(y_true)))
	# K.epsilon() is the fuzz factor, included
	# to ensure that division by 0 doesn't happen
	return (1 - mse/(var + K.epsilon()))

"""
Adjusted R^2
Mathematically, adjusted R^2 is given by:
                _                   _
               | (1 - R^2) * (n - 1) |
(R^2)adj = 1 - |---------------------|
               |_     n - k - 1     _|
where n is the number of samples and k is the number
of features per sample. Where R^2 will increase with
the number of samples and with the number of features,
adjusted R^2 will increase with the numbers of USEFUL
samples and features. This makes it much better for
the requested exploration of what altering the number
of input features does to model accuracy.
Unfortunately, getting the number of input features
from y_true (the ground truth data) and y_pred (the
model's predictions) is impossible. Getting the
number of samples from the same should be possible,
but everything we tried gave odd or unexpected
results, and we couldn't figure out how to do it.
We also couldn't figure out how to pass those values
as parameters to the function, as Keras requires
that loss functions and metrics have only y_true
and y_pred as inputs. Thus, in the below function,
they are manually defined. n is chosen to be 23,
the number of samples in the test dataset, as the
model's performance on the data it hasn't been trained
on is the most relevant metric for its performance.
k is manually updated every time the number of features
is changed. Changing the number of features in code
where this is called without also changing k here may
provide strange or otherwise inaccurate results.
"""

def adjusted_r2(y_true, y_pred):
	r2 = coeff_determination(y_true, y_pred)
	n = 23
	k = 3
	z = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))
	return z
