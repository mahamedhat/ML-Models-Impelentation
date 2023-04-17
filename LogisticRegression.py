import numpy as np


def sigmoid (x):
    return (1/(1+np.exp(-x)))
class LogisticRegression:

    def __init__(self,alpha = 0.001, n_iters = 1000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        m , n = X.shape 
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iters):
            # model
            
            linear_pred = np.dot(self.weights,X.T) +self.bias
            y_pred = sigmoid(linear_pred)
            # gradients 
            dj_dw = (1/m) * np.dot((y_pred- y),X)
            dj_db = (1/m) * np.sum(y_pred- y)

            self.weights -= self.alpha * dj_dw
            self.bias -= self.bias * dj_db







    def predict(self, X):
        linear_pred = np.dot(self.weights,X.T) +self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<0.5 else 1 for y in y_pred]
        return class_pred
    


    
    def accuracy(self,y_test , y_pred):
        return np.sum(y_pred == y_test)/len(y_test)

