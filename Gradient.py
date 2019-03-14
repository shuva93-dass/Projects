import numpy as np
import pandas as pd

#Importing the training dataset
dataset = pd.read_csv('carbig.csv');

#Imputing median for nan values
dataset = dataset.fillna(dataset.median())


X_train = dataset.iloc[:, 0:1].values #predictor
T = dataset.iloc[:, 1:2].values #target

#Converting predictor into a matrix
o = np.array(X_train)
X = np.array(X_train)
k = np.max(X[:,0])
X = X/k
# adding 1 to Predictor
#print(X)
X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

#Extracting max value along the column to normalize
X_t = np.transpose(X)
#
# target
t = np.matrix(T)
t_T= t.T

# random values for initial weights
np.random.seed(600)
cur_w = np.random.randn(1,2)
cur_w = np.matrix(cur_w)
print (cur_w)

# gradient of loss function
h = X.T@X
def J(w):
    return (2*w*h)  - (2*t_T*X)


rate = 1e-3
max_iters = 5000 # maximum number of iterations
iters = 0
prev_w = np.matrix([0,0])
while not np.allclose(prev_w, cur_w) and iters < max_iters:
    prev_w = cur_w #Store current w value in prev_w
    cur_w = cur_w - rate * J(prev_w)  #Grad descent
    iters = iters+1 #iteration count
    print("Iteration",iters,"\nW value is",cur_w) #Print iterations

#Denormalizing weights
w_1 = cur_w[0,0]/k #w1
w_0 = cur_w[0,1] #w0 remains unchanged

#Denormalized Final weight vector
fin_w = np.matrix([w_1,w_0])

#Prediction: y = w.x
q = np.hstack((o, np.ones((o.shape[0], 1), dtype= o.dtype)))
y = fin_w  * q.T
y_t = np.transpose(y)

#Plotting
import matplotlib.pyplot as plt
plt.scatter(X_train, T, label= "stars", color= "green",
            marker= "*", s=30)
plt.title('gradient')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.plot(X_train,y_t,linewidth=2.0)
plt.show()


