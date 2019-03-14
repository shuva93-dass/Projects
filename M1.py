#import random
import numpy as np
from numpy import linalg as LA

# #samples = 10
np.random.seed(45)

ep = np.random.normal(0, 0.3,(1,10))
x_train = np.random.uniform(0,1,10)
y_train = np.sin(x_train * np.pi*2.) + ep


ep = np.random.normal(0, 0.3,(1,100))
x_test = np.random.uniform(0,1,100)
y_test = np.sin(x_test * np.pi*2.) + ep


x_testM = np.matrix(x_test).T
x_trainM = np.matrix(x_train).T
y_trainM = np.matrix(y_train).T
y_testM = np.matrix(y_test).T
N = x_trainM.shape[0]

one = np.ones((x_trainM.shape[0], 1))
phi = np.c_[one,np.power(x_trainM,1),np.power(x_trainM,2),np.power(x_trainM,3),np.power(x_trainM,4),np.power(x_trainM,5),np.power(x_trainM,6),np.power(x_trainM,7),np.power(x_trainM,8),np.power(x_trainM,9)]
E_train = []  
weight = []     
for i in range(10):
    k = phi[:,0:i+1].T@phi[:,0:i+1]
    #COND = np.linalg.cond(k)
    inv =  np.linalg.pinv(k)
    r = inv*phi[:,0:i+1].T
    w = r*y_trainM
    w = w.T
    w = np.matrix(w)
    weight.append(w)
    #Predicted y
    train_y =  w* phi[:,0:i+1].T
    #L2 norm
    diff = y_trainM - train_y.T
    J_train = np.square(LA.norm(diff))
    #Erms
    E = np.sqrt(J_train/N)
    E_train.append(E)

print(E_train)
print (weight)   
 
S = x_testM.shape[0]
one = np.ones((x_testM.shape[0], 1))
phi = np.c_[one,np.power(x_testM,1),np.power(x_testM,2),np.power(x_testM,3),np.power(x_testM,4),np.power(x_testM,5),np.power(x_testM,6),np.power(x_testM,7),np.power(x_testM,8),np.power(x_testM,9)]
E_test = []       
for i in range(10):
    
    test_y =  weight[i]* phi[:,0:i+1].T
    #L2 norm
    diff = y_testM - test_y.T
    J_test = np.square(LA.norm(diff))
    #Erms
    E = np.sqrt(J_test/S)
    E_test.append(E)

print(E_test)      
M =[] 
for i in range(10):
    M.append(i)
import matplotlib.pyplot as plt
#plt.scatter(M, E_train)
plt.plot(M, E_train, '-ok',color='blue')
plt.plot(M, E_test, '-ok',color='red')
plt.xlabel('Model Complexity')
plt.ylabel('Erms')
plt.legend(["Train error","Test Error"])
plt.title("Error for sample = 10")
#plt.xlim([0,10])
#plt.ylim([0,50])
plt.show()


# #samples = 100

ep = np.random.normal(0, 0.3,(1,100))
x_train = np.random.uniform(0,1,100)
y_train = np.sin(x_train * np.pi*2.) + ep

ep = np.random.normal(0, 0.3,(1,100))
x_test = np.random.uniform(0,1,100)
y_test = np.sin(x_test * np.pi*2.) + ep


x_testM = np.matrix(x_test).T
x_trainM = np.matrix(x_train).T
y_trainM = np.matrix(y_train).T
y_testM = np.matrix(y_test).T
N = x_trainM.shape[0]

one = np.ones((x_trainM.shape[0], 1))
phi = np.c_[one,np.power(x_trainM,1),np.power(x_trainM,2),np.power(x_trainM,3),np.power(x_trainM,4),np.power(x_trainM,5),np.power(x_trainM,6),np.power(x_trainM,7),np.power(x_trainM,8),np.power(x_trainM,9)]
E_train = []  
weight = []     
for i in range(10):
    k = phi[:,0:i+1].T@phi[:,0:i+1]
    #COND = np.linalg.cond(k)
    inv =  np.linalg.pinv(k)
    r = inv*phi[:,0:i+1].T
    w = r*y_trainM
    w = w.T
    w = np.matrix(w)
    weight.append(w)
    #Predicted y
    train_y =  w* phi[:,0:i+1].T
    #L2 norm
    diff = y_trainM - train_y.T
    J_train = np.square(LA.norm(diff))
    #Erms
    E = np.sqrt(J_train/N)
    E_train.append(E)

print(E_train)
print (weight)   
 
S = x_testM.shape[0]
one = np.ones((x_testM.shape[0], 1))
phi = np.c_[one,np.power(x_testM,1),np.power(x_testM,2),np.power(x_testM,3),np.power(x_testM,4),np.power(x_testM,5),np.power(x_testM,6),np.power(x_testM,7),np.power(x_testM,8),np.power(x_testM,9)]
E_test = []       
for i in range(10):
    
    test_y =  weight[i]* phi[:,0:i+1].T
    #L2 norm
    diff = y_testM - test_y.T
    J_test = np.square(LA.norm(diff))
    #Erms
    E = np.sqrt(J_test/S)
    E_test.append(E)

print(E_test)      
M =[] 
for i in range(10):
    M.append(i)
import matplotlib.pyplot as plt
#plt.scatter(M, E_train)
plt.plot(M, E_train, '-ok',color='blue')
plt.plot(M, E_test, '-ok',color='red')
plt.xlabel('Model Complexity')
plt.ylabel('Erms')
plt.legend(["Train error","Test Error"])
plt.title("Error for sample = 100")
#plt.xlim([0,10])
#plt.ylim([0,30])
plt.show()





