import numpy as np
from numpy import exp  # exponential function.
from numpy import log  # logarithm function.
from numpy import dot  # dot product.
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class MyLogisticRegression:

    def __init__(self, lr=0.001, max_iter=1000, tolerance=1e-4):
        self.weights = None
        self.bias = None
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance = tolerance

    def sigmoid(self, x):
        x[x < -709] = -709
        return 1 / (1 + exp(-x))

    def fitted_val(self, X):
        linear = dot(X, self.weights) + self.bias
        fitted_val = self.sigmoid(linear)
        return fitted_val

    def predict(self, X):
        fitted_val = self.fitted_val(X)
        pred = np.zeros(X.shape[0])
        pred[fitted_val >= 0.5] = 1
        return pred

    def loss(self, X, y):
        pred = self.fitted_val(X)
        loss_vec = y * log(pred) + (1 - y) * log(1 - pred)
        loss = -np.sum(loss_vec) / len(pred)
        return loss

    def gradient_weights(self, X, y):
        pred = self.fitted_val(X)
        grad = dot(pred - y, X) / len(pred)
        return grad

    def gradient_bias(self, X, y):
        pred = self.fitted_val(X)
        grad = np.sum(pred - y) / len(pred)
        return grad

    def fit(self, X, y):
        # init values.
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        # gradient descent algo.
        converged = False
        for _ in range(self.max_iter):
            gradient_weights = self.gradient_weights(X, y)
            gradient_bias = self.gradient_bias(X, y)
            new_weights = self.weights - self.lr * gradient_weights
            new_bias = self.bias - self.lr * gradient_bias
            if np.sum((new_weights - self.weights) ** 2) + (new_bias - self.bias) ** 2 < self.tolerance:
                converged = True
                break
            self.weights = new_weights
            self.bias = new_bias
        if not converged:
            print("Failed to Converge.")


# testing.
df = datasets.load_breast_cancer()
X, y = df.data, df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
model = MyLogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
my_pred = model.predict(X_test)
print("length is: ", len(my_pred))
print(my_pred)
my_accuracy = np.mean(my_pred == y_test)
print(my_accuracy)

# logistic model from sklearn.
clf = LogisticRegression(max_iter=10000, random_state=123)
clf = clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(pred)
accuracy = np.mean(pred == y_test)
print(accuracy)

# compare
print("diff = ", np.sum(my_pred != pred))
