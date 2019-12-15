import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D


filename = ("housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(filename, delim_whitespace=True, names=names)
df = df.drop(['CRIM' ,'ZN' ,'INDUS' ,'NOX' ,'AGE' ,'DIS' ,'RAD', 'CHAS' ,'PTRATIO' ,'TAX' ,'B'], axis = 1)
df.head(5)

x = df.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dataset = pd.DataFrame(x_scaled)
dataset.head(5)


x1 = dataset[0].values
x2 = dataset[1].values
Y= dataset[2].values

m = len(x1)
x0 = np.ones(m)
X = np.array([x0, x1, x2]).T    # .T is used to obtain transpose
B = np.zeros(3)
Y = np.array(Y)
alpha = 0.0001      # alpha is learning rate


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J


inital_cost = cost_function(X, Y, B)


def grad_desc(X, Y, B, a, iteration):
    cost_iter = [0] * iteration
    m = len(Y)

    for i in range(iteration):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - a * gradient
        cost = cost_function(X, Y, B)
        cost_iter[i] = cost

    return B, cost_iter


newB, cost_history = grad_desc(X, Y, B, alpha, 200000)
Ypred = X.dot(newB)

t3D = plt.figure().gca(projection='3d')
t3D.scatter(x1, x2, Y)
t3D.scatter(x1, x2, Ypred)
t3D.set_xlabel('LSTAT')
t3D.set_ylabel('RM')
t3D.set_zlabel('MEDV')
plt.figure(figsize=(500, 400))
plt.show()

x = np.arange(1, 200001)
plt.plot(x, cost_history)
plt.xlabel('iteration')
plt.ylabel('cost function')
plt.show()


def normalize(Y, Y_pred):
    normalize = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return normalize


print(normalize(Y, Ypred))