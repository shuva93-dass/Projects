#  MACHINE LEARNING Projects

ClosedForm.py:
This file is an implementation of linear regression on a dataset using Closed-Form Solution.

Gradient.py:
This file is an implementation of linear regression on a dataset using Gradient Descent Method.

M1.py:
Generated synthetic training and testing dataset from uniform and gaussian distribution respectively and used the method of linear regression with non-linear models to fit polynomials of degree M(complexity) = 0,1,2,....,9 to the training set and recorded the training and testing errors for each of the 10 cases.From the graph  between complexity of the model vs train error and test error, it was observed that the as complexity of the model increases, the train error decreases while the test error shoots up towards the end thereby  increasing the gap between the train error and test error. Its a clear indication of overfitting.

Solution3.py:
In this file, I again used the same dataset generated from the previous file,sampled 100 datasets out of the distribution, each containing 25 observations  but this time  used regularization factor 𝜆 to reduce the test error and selected a set of permissible limits of  𝜆 and for each value of 𝜆, use the method of “linear regression with non-linear models” to fit Gaussian basis functions to each of the datasets.Plotted bias_square and variance against the complexity of the model. I observed that as we increase lamda, the variance of the model decreases, bias increases and the test error first decreases and then increases.
