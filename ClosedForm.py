import numpy as np
import pandas as pd

#Importing the training dataset
dataset = pd.read_csv('carbig.csv')
#Imputing 0 for nan values
dataset = dataset.fillna(dataset.median())
#dataset = dataset.fillna(0)

#Separating predictor and target
X_train = dataset.iloc[:, 0:1].values
Y_train = dataset.iloc[:, 1:2].values

#Converting predictor into a matrix
o = np.matrix(X_train)
nu = np.matrix(X_train)
h= np.max(nu[:,0])
nu = nu/h

#Predictor
X = np.hstack((nu, np.ones((nu.shape[0], 1), dtype=nu.dtype)))


X_t = np.transpose(X)
#X_t = np.transpose(X)

# target
t = np.matrix(Y_train)
#Closed Form Solution
k = X.T@X
inv = k.I
r =inv*X.T
#weight
w = r*t
w = w.T
w = np.matrix(w)
w_1 = w[0,0]/h #w1
w_0 = w[0,1] #w0 remains unchanged

#Denormalized Final weight vector
fin_w = np.matrix([w_1,w_0])

# Estimated target valies
#p = w[0,0]*h
#w = np.matrix([p,w[1,0]])


#y = np.transpose(w)*X_t
q = np.hstack((o, np.ones((o.shape[0], 1), dtype= o.dtype)))
y = fin_w * q.T
y_t = np.transpose(y)
#nu = nu*h
#Graph
import matplotlib.pyplot as plt
plt.scatter(X_train, Y_train, label= "stars", color= "green",  
            marker= "*", s=30) 
plt.title('ClosedFormSolution')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.plot(X_train,y_t,linewidth=2.0)
plt.legend(["ClosedForm"])
plt.show()


#z = np.ones((406,1))
#np.append(nu, z, axis=1)
#nu
#dataset[1].fillna(0, inplace=True)

