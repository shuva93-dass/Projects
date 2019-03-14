import numpy as np
from numpy import linalg as LA


np.random.seed(45)

x_tr = np.random.uniform(0,1,25)
X = np.linspace(x_tr.min(), x_tr.max(), 25, endpoint=True)
y_tr = np.sin(X * np.pi*2.)
y_trM = np.matrix(y_tr).T


lam = [0.049,0.135,0.367,1,2.718,7.389]
func = []
fx_avg = []
bias_sq = []
variance = []
E_average = []
for item in lam:
    fx = []
    sum=0
    E_add = 0
    E_fx = []
    for D in range(0,100):

        #train data
        ep = np.random.normal(0, 0.3,(1,25))
        x_train = np.random.uniform(0,1,25)
        X_tr  = np.linspace(x_train.min(), x_train.max(), 25, endpoint=True)
        y_train = np.sin(X_tr * np.pi*2.) + ep

        x_trainM = np.matrix(X_tr).T
        y_trainM = np.matrix(y_train).T
        
        #test data
        ep = np.random.normal(0, 0.3,(1,1000))
        x_test = np.random.uniform(0,1,1000)
        X_tr_test  = np.linspace(x_test.min(), x_test.max(), 1000 , endpoint=True)
        y_test = np.sin(X_tr_test * np.pi*2.) + ep

        x_testM = np.matrix(X_tr_test).T
        y_testM = np.matrix(y_test).T

        ss = np.square(0.1)
        M = 5
        one = np.ones((x_trainM.shape[0], 1))
        mean_list=[0.15,0.25,0.5,0.75,0.9]
        
        def dist(i,N,x_matrix):
           mean = np.matrix(np.repeat(mean_list[i-1],N))
           g = np.power((x_matrix - mean.T),2)
           g = g/(2*ss)
           gauss = np.exp(-g)
           return gauss

        #train
        phi = one
        for i in range(1,M+1):
            phi = np.concatenate((phi,dist(i,25,x_trainM)),axis=1)
        
        k = phi.T@phi 
        k = k + (item*np.identity(M+1))
        inv =  np.linalg.pinv(k)
        r = inv*phi.T
        w = r*y_trainM
        w = w.T
        w = np.matrix(w)
    
        y = w * phi.T  
  
        y_t = np.transpose(y)
        sum+=y_t
        fx.append(y_t)
        
        # Test error
        one_test = np.ones((x_testM.shape[0], 1))
        phi_test = one_test
        for i in range(1,M+1):
            phi_test = np.concatenate((phi_test,dist(i,1000,x_testM)),axis=1)
        
        y_test = w * phi_test.T
        diff = y_testM - y_test.T
        J_test = np.square(LA.norm(diff))
        #Erms
        E = np.sqrt(J_test/1000)
        E_fx.append(E)
        E_add+=E
        
        if (D == 99):
           func.append(fx) 
           #fx_avg
           avg = sum/100
           fx_avg.append(avg)
           #bias
           b = np.power((avg - y_trM),2)
           b_avg = np.sum(b,axis=0)/25
           bias_sq.append(b_avg)
           #variance
           add=0
           for i in range(100):
              add+=np.square(fx[i] - avg)
           v = add/100
           var= (np.sum(v,axis=0))/25
           variance.append(var) 
           #avg_test_error
           E_avg = E_add/100
           E_average.append(E_avg)
           

import matplotlib.pyplot as plt

"""
import math
j=0  
for f in func:
    item = lam[j]
    ln = math.log(item)
    for i in range(0,100):
            plt.plot(X, y_tr , color = 'blue') 
            plt.plot(X,f[i], color = 'red',alpha=0.2)
            plt.title('ln %s = %d' %(item,ln))
            #plt.title('graph')
            plt.xlabel('Predictor')
            plt.ylabel('Target')
    plt.show()
    j+=1

for i in range(6):
    plt.plot(X, y_tr , color = 'blue') 
    plt.plot(X,fx_avg[i], color = 'red')
    plt.show()

"""

log_lam = np.log(lam)

for i in range(6):
    bias_sq[i] = bias_sq[i].tolist()
    variance[i] = variance[i].tolist()
    
bias_sq=[i[0] for i in bias_sq]
bias_sq=[i[0] for i in bias_sq]

variance=[i[0] for i in variance]
variance=[i[0] for i in variance]

r=[]
for i in range(6):
    r.append(bias_sq[i] + variance[i])

plt.plot(log_lam, bias_sq , color = 'blue') 
plt.plot(log_lam, variance , color = 'red')
plt.plot(log_lam, r, color = 'magenta')
plt.plot(log_lam, E_average, color = 'black')
plt.xlabel("ln"+u"\u03BB")
plt.legend(["bias_square","variance","bias_square+variance","test_error"])
plt.show()
    

