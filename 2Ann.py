import numpy as np
import matplotlib.pyplot as plt 

def sigmoiod(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def tanh(x):
    return np.tanh(x)
def softmax(x):
    ex=np.exp(x-np.max(x))
    return ex/ex.sum()

x=np.linspace(-5,5,100)

y_sig=sigmoiod(x)
y_relu=relu(x)
y_tan=tanh(x)
y_sm=softmax(x)

plt.subplot(2,2,1)
plt.plot(x,y_sig)
plt.xlabel("x")
plt.ylabel('f(x)')

plt.subplot(2,2,2)
plt.plot(x,y_relu)
plt.xlabel("x")
plt.ylabel('f(x)')