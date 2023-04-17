import numpy as np

class LinearRegression:

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
            y_pred = np.dot(self.weights,X.T) +self.bias
            # gradients 
            dj_dw = (1/m) * np.dot((y_pred- y),X)
            dj_db = (1/m) * np.sum(y_pred- y)

            self.weights -= self.alpha * dj_dw
            self.bias -= self.bias * dj_db







    def predict(self, X):
        y_pred = np.dot(self.weights,X.T) +self.bias
        return y_pred
    


    
    def mse(self,y_test , y_pred):
        return np.mean((y_test - y_pred)**2)

