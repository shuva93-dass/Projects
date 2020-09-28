#from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numpy import linalg as LA

class Neural_Network(object):
    def __init__(self,ounits,hunits,X):        
        self.inputLayerSize = X.shape[1]
        self.outputLayerSize = ounits
        self.hiddenLayerSize = hunits
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.B1 = np.random.randn(1)
        self.B2 = np.random.randn(1)
        
    def forwardfeed(self, X):
        self.a1 =  np.dot(X, self.W1) 
        self.z1 = self.sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.W2) + self.B2
        yHat = self.sigmoid(self.a2) 
        return yHat
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
   
    def sigmoidDerivative(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        self.yHat = self.forwardfeed(X)
        J = 0.5*((y-self.yHat)**2)
        return J
        
    def gradient(self, X, y):
        self.yHat = self.forwardfeed(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidDerivative(self.a2))
        #delta3 = -(y-self.yHat)
        dJdW2 = np.dot(self.z1.T, delta3) 
        
        delta2 = np.multiply((np.matmul(delta3 , self.W2.T)),self.sigmoidDerivative(self.a1))

        dJdW1 = np.dot(X.T, delta2) 
        
        dJdB2 = np.multiply(self.B2, np.mean(delta2))
         
        return dJdW1, dJdW2,dJdB2
np.random.seed(45)
# INPUT
X = np.array([[0,0,1], [1,0,1], [0,1,1],[1,1,1]])
#OUTPUT
y = np.array([[1,0], [0,1], [0,1],[1,0]])
NN = Neural_Network(2,2,X)
rate = 1.8
max_iters = 100000 #maximum number of iterations
iters = 0 
err = 1
costJ = []
#get_ipython().run_line_magic('matplotlib', 'qt')
plt.figure(1)   
while (err > 0.001  and iters < max_iters):
        old_w1 = NN.W1   
        old_w2 = NN.W2   
        old_b2 = NN.B2 
        
        yHat = NN.forwardfeed(X)
        cost = NN.costFunction(X,y)
        costJ.append(np.mean(cost))
        
        rem = iters % 5000
        
        if(rem == 0):
            lines = plt.plot(costJ)#,marker = '.')
            plt.setp(lines, color='r',linewidth = 0.5)
            plt.title('Cost = %f'%(np.mean(cost)))
            plt.xlabel('Iterations')
            plt.ylabel('Cost')
            plt.draw()
            plt.pause(1e-17)
            time.sleep(0.1)
            
        err = LA.norm(y - yHat)
        #print (err)
        
        #gradient descent
        dJdW1, dJdW2, dJdB2 = NN.gradient(X,y) 
        NN.W1 = old_w1 - rate * dJdW1
        NN.W2 = old_w2 - rate * dJdW2
        NN.B2 = old_b2 - rate * dJdB2
        
        iters = iters+1
   

plt.show()
plt.pause(3)
   
    
#TestData
plt.figure(2)
x2 = np.arange(0,1.05,0.05)
x2_v = np.array([],dtype=np.float64)


for i in range(x2.shape[0]):
   x2_v = np.hstack((x2,x2_v))

x1 = x2_v.copy()
x1.sort()

X_test = np.c_[x1,x2_v,np.ones(x2_v.shape[0])]


NN.W1 = old_w1
NN.W2 = old_w2

y_test = np.round(NN.forwardfeed(X_test))

#plot 2D
#get_ipython().run_line_magic('matplotlib', 'qt')

for i in range(y_test.shape[0]):
    if y_test[i,0] == 1:
        plt.plot(X_test[i,0],X_test[i,1],'og') 
    else:
        plt.plot(X_test[i,0],X_test[i,1],'*b') 

plt.plot(-1,1,'ks',1,1,'ks',-1,-1,'ks',1,-1,'ks')
plt.title('Decision Boundary')
plt.xlim(0.5 ,1.5)
plt.ylim(0.5 ,1.5)


#plt.show()

#plot 3D
#get_ipython().run_line_magic('matplotlib', 'qt')
plt.figure(3)
NN.W1 = old_w1
NN.W2 = old_w2
xx1, xx2 = np.meshgrid(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1).T)
z = np.zeros((xx1.shape[0],xx1.shape[0]))
for i in range(xx1.shape[1]):
    x2_test = np.c_[xx1[:,i],xx2[:,i],np.ones(xx1[:,i].shape[0])]
    y2_test = np.round(NN.forwardfeed(x2_test))
    z[:,i] = y2_test[:,0]
    
    #Axes3D.plot_surface(xx1[:,i],xx2[:,i],z[:,i], norm=None, vmin=None, vmax=None, lightsource=None)

ax = plt.axes(projection='3d')

ax.plot_surface(xx1,xx2,z ,rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('3d Visualisation');
plt.draw()
plt.pause(10)

