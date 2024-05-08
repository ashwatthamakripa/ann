import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, lr=0.1, n_iter=100):
        self.lr = lr # learning rate
        self.n_iter = n_iter 
        self.weights = None 
        self.bias = None 
        
    def fit(self, X, y):
         
        self.weights = np.zeros(X.shape[1])
        self.bias = 1

        
        for i in range(self.n_iter):
           
            for j in range(X.shape[0]):
                
                pred = self.predict(X[j])
             
                if pred != y[j]:
                    self.weights += self.lr * y[j] * X[j]
                    self.bias += self.lr * y[j]

    def predict(self, X):
       
        output = np.dot(X, self.weights) + self.bias
        
        return np.where(output >= 0, 1, -1)
    
X = np.array([[2, 2], [4, 4], [4, 0], [3, 2], [8, 4], [8, 0]])
y = np.array([1, 1, -1, 1, -1, -1])

model = Perceptron()
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlim(0, 10)
plt.ylim(0, 5)
x1 = np.linspace(0, 10)
x2 = -(model.weights[0]*x1 + model.bias)/model.weights[1]
plt.plot(x1, x2, '-r')
plt.show()
