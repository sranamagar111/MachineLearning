from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Loading Boston House Price DataSet
boston_dataset = load_boston()
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
df['PRICE'] = boston_dataset.target

X = df.iloc[:, 6]
y = df.iloc[:, 13]


# Defining the hypothesis function
# h(x) = theta_0 + theta_1*x
def hypothesis(theta_0, theta_1, x):
    return theta_0 + (theta_1 * x)


# Calculating the error of the line
def cost(theta_0, theta_1, X, y):
    cost_value = 0
    for (xi, yi) in zip(X, y):
        cost_value += (0.5 * ((hypothesis(theta_0, theta_1, xi) - yi) ** 2))
    return cost_value


# Calculating the derivatives
# (i.e. line that has the lowest error)
def derivatives(theta_0, theta_1, X, y):
    der_theta_0 = 0
    der_theta_1 = 0
    for (xi, yi) in zip(X, y):
        der_theta_0 += hypothesis(theta_0, theta_1, xi) - yi
        der_theta_1 += (hypothesis(theta_0, theta_1, xi) - yi) * xi

    der_theta_0 /= len(X)
    der_theta_1 /= len(X)
    return der_theta_0, der_theta_1


# Updating our parameter to reduce the gradient value
def perform_gradient_descent(theta_0, theta_1, X, y, alpha):
    der_theta_0, der_theta_1 = derivatives(theta_0, theta_1, X, y)
    theta_0 = theta_0 - (alpha * der_theta_0)
    theta_1 = theta_1 - (alpha * der_theta_1)
    return theta_0, theta_1


# Linear regression function implementation
def linear_regression(X, y):
    theta_0 = 1
    theta_1 = 0.1

    for i in range(0, 1000):
        theta_0, theta_1 = perform_gradient_descent(theta_0, theta_1, X, y, 0.005)
    print(np.sqrt(sum((X - y) ** 2) / len(X)))

linear_regression(X, y)
