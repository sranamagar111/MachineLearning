
'''
    ML Classroom Assignment 1
    Brihat Ratna Bajracharya
    19/075
    CDCSIT
'''



# ALL IMPORTS HERE

import pandas as pd 
import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import seaborn as sns

# VISUALIZATION (QUESTION 1)


# read csv

data = pd.read_csv('housing.csv', delim_whitespace=True)
data

# correlation test among feature columns with MEDV column

corr = data.corr()
print(corr['MEDV'])

labelcorr = corr['MEDV'].sort_values()
corr_dict = dict(labelcorr)

# sort in descending order of absolute value of correlation coefficient

sorted(corr_dict, key=lambda dict_key: abs(corr_dict[dict_key]), reverse=True)

# obtain data for most correlated values

lstat = data['LSTAT']
room = data['RM']
pt_ratio = data['PTRATIO']

price = data['MEDV']

# lSTAT vs MEDV plot

plt.plot(lstat, price, 'g+')
plt.xlabel('% Lower Status of Population')
plt.ylabel('Price of house (in 1000s)')

plt.show()

# RM vs MEDV plot

plt.plot(room, price, 'b+')
plt.xlabel('Avg number of rooms (per house)')
plt.ylabel('Price of house (in 1000s)')

plt.show()

# PTRATIO vs MEDV plot

plt.plot(pt_ratio, price, 'r+')
plt.xlabel('Pupil-Teacher Ratio')
plt.ylabel('Price od house (in 1000s)')

plt.show()

# train test split using sklearn

' get X and Y '
Y = data[list(data.columns)[-1]]

selected_columns = ['LSTAT']

X = data[selected_columns]
X_train_lstat, X_test_lstat, Y_train_lstat, Y_test_lstat = train_test_split(X, Y, test_size =0.2)

X = room
X_train_room, X_test_room, Y_train_room, Y_test_room = train_test_split(X, Y, test_size =0.2)

X = pt_ratio
X_train_ptratio, X_test_ptratio, Y_train_ptratio, Y_test_ptratio = train_test_split(X, Y, test_size =0.2)

print(X_train_lstat)
# print(X_test_room)
print(Y_train_lstat)
# print(Y_test_room)


# plot train set data (lstat)

X_train = X_train_lstat
Y_train = Y_train_lstat

plt.plot(X_train, Y_train, 'g*')
plt.xlabel('% Lower status of popn (train set)')
plt.ylabel('Price of house (in 1000s, train set)')

plt.show()

# plot train set data (rm)

X_train = X_train_room
Y_train = Y_train_room

plt.plot(X_train, Y_train, 'b*')
plt.xlabel('Avg num of rooms per house (train set)')
plt.ylabel('Price of house (in 1000s, train set)')

plt.show()

# plot train set data (pt_ratio)


X_train = X_train_ptratio
Y_train = Y_train_ptratio

plt.plot(X_train, Y_train, 'r*')
plt.xlabel('Pupil-Teacher ratio (train set)')
plt.ylabel('Price of house (in 1000s, train set)')

plt.show()

# LINEAR REGRESSION (QUESTION 2)


# change here for different linear regression

X_train = X_train_lstat
X_test = X_test_lstat
Y_train = Y_train_lstat
Y_test = Y_test_lstat


# X_train = X_train_room
# X_test = X_test_room
# Y_train = Y_train_room
# Y_test = Y_test_room

# X_train = X_train_ptratio
# X_test = X_test_ptratio
# Y_train = Y_train_ptratio
# Y_test = Y_test_ptratio


# cost function

def costFunction(xVector, yVector, theta):
    inner = np.power(((xVector * theta.T) - yVector), 2)
    return np.sum(inner) / 2


# pre for linear regression

array_ones = np.ones(len(X_train))
# print(array_ones)
xVector = np.column_stack((array_ones, X_train))
# print(xVector)

# yVector = np.matrix(data['MEDV']).T
yVector = np.matrix(Y_train).T

# print(yVector)

theta = np.matrix(np.array([0.00, 0.00]))
# print(theta.T)


# Linear Regression from scratch

learningRate = 0.0001
iterations = len(X_train)

costs = np.zeros(iterations)

m = np.size(theta,1 )

newTheta = theta.T

# print(newTheta)

for iter in range(iterations):
    costs[iter] = costFunction(xVector, yVector, theta)

    for i in range(len(xVector)):
        currentError = yVector[i,0 ] - (xVector[i,: ] * newTheta)
        for j in range(m):
            term = np.multiply(np.multiply(currentError, xVector[i,j ]), learningRate)

            newTheta[j,0 ] = newTheta[j,0 ] + term
    #         print(i, j, newTheta)
print(newTheta)
# print(costs)


# linear regressoin on actual data

t0 = float(newTheta[0])
t1 = float(newTheta[1])

plt.plot(X_train,Y _train,' g+')
# plt.plot(X_train,t0 + t1*X_train,'r*')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = t0 + t1 * x_vals
plt.plot(x_vals, y_vals, 'r--')
plt.xlabel('% Lower status of popn (train set)')
plt.ylabel('Price of house (in 1000s, train set)')

plt.show()

'''
# evaluation of test dataset

array_ones_test = np.ones(len(X_test))
# print(array_ones_test)

xVectorTest = np.column_stack((array_ones_test, X_test))
# print(xVectorTest)

y_test_predict = xVectorTest * newTheta
# print(y_test_predict)

yVectorTest = np.matrix(Y_test).T
# print(yVectorTest)

plt.plot(X_test,Y_test,'g+')
# plt.plot(X_test,t0 + t1*X_test,'r*')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = t0 + t1 * x_vals
plt.plot(x_vals, y_vals, 'r--')

plt.show()
'''


# rmse of test set

# np.sqrt(mean_squared_error(yVectorTest, y_test_predict))


# Normalize Part from here (QUESTION 3)


# Normalizing function
def normalize(X):
    mins = np.min(X, axis = 0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X) / rng)
    return norm_X


# xdata = data['LSTAT']
# xdata_norm = normalize(xdata)


X_train_norm = normalize(X_train)
# X_train_norm
Y_train_norm = normalize(Y_train)
# Y_train_norm
X_test_norm = normalize(X_test)
# X_test_norm
Y_test_norm = normalize(Y_test)
# Y_test_norm


# pre for linear regression (normalize)

array_ones_norm = np.ones(len(X_train_norm))
# print(array_ones_norm)

xVectorNorm = np.column_stack((array_ones_norm, X_train_norm))
# print(xVectorNorm)

yVectorNorm = np.matrix(Y_train_norm).T

# print(yVectorNorm)

theta_norm = np.matrix(np.array([0.00, 0.00]))
# print(theta_norm.T)


# Linear Regression from scratch (normalize)
learningRate = 0.0001
iterations = len(X_train_norm)

costsNorm = np.zeros(iterations)

mNorm = np.size(theta_norm, 1)

newThetaNorm = theta_norm.T

# print(newTheta)

for iter in range(iterations):
    costsNorm[iter] = costFunction(xVectorNorm, yVectorNorm, theta_norm)

    for i in range(len(xVectorNorm)):
        currentErrorNorm = yVectorNorm[i, 0] - (xVectorNorm[i, :] * newThetaNorm)
        for j in range(mNorm):
            termNorm = np.multiply(np.multiply(currentErrorNorm, xVectorNorm[i, j]), learningRate)

            newThetaNorm[j, 0] = newThetaNorm[j, 0] + termNorm
    #       print(i, newThetaNorm)
print(newThetaNorm)
# print(costs)


# linear regressoin on actual data (normalize)

t0 = float(newThetaNorm[0])
t1 = float(newThetaNorm[1])

plt.plot(X_train_norm, Y_train_norm, 'g+')
# plt.plot(X_train_norm,t0 + t1*X_train_norm,'r*')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = t0 + t1 * x_vals
plt.plot(x_vals, y_vals, 'r--')
plt.xlabel('% Lower status of popn (normalized train set)')
plt.ylabel('Price of house (in 1000s, normalized train set)')

plt.show()

'''
#evaluation of test dataset (normalize)

array_ones_test_norm = np.ones(len(X_test_norm))
# print(array_ones_test_norm)

xVectorTestNorm = np.column_stack((array_ones_test_norm, X_test_norm))
# print(xVectorTest)

y_test_predict_norm = xVectorTestNorm * newThetaNorm
# print(y_test_predict)

yVectorTestNorm = np.matrix(Y_test_norm).T
# print(yVectorTestNorm)

plt.plot(X_test_norm,Y_test_norm,'g+')
# plt.plot(X_test_norm,t0 + t1*X_test_norm,'r*')

axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = t0 + t1 * x_vals
plt.plot(x_vals, y_vals, 'r--')

plt.show()
'''

# rmse of test set (normalize)


# np.sqrt(mean_squared_error(yVectorTest, y_test_predict))


# logistic Regression from here (QUESTION 4)


# read csv

data = pd.read_csv('Logisticdataset.csv', delimiter=',')
data

X = data[data.columns[:-1]].to_numpy()
# print(X)
y = data[data.columns[-1]].to_numpy()


# print(y)


# adds 1 to beginning of X vector
def add_ones(X):
    array_ones = np.ones((X.shape[0], 1))
    return np.concatenate((array_ones, X), axis=1)


XX = add_ones(X)
# XX


# plotting x1, x2 vs y

plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend();


# sigmoid function

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# loss function

def lossfunction(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# function that calculates logistic regression conefficients

def myLogisticRegression(X, verbose=False):
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    iterations = len(X)

    for i in range(iterations):
        z = np.dot(X, theta)
        h = sigmoid(z)

        gradient = np.dot(X.T, (h - y)) / y.size

        theta -= learning_rate * gradient
        #         print(theta)

        if (verbose == True and i % iterations == 0):
            #             print(f'loss: {lossfunction(h, y)} \t')
            print(theta)
    #             pass
    return theta

% time
theta_calc = myLogisticRegression(XX, True)

theta_calc

plt.figure(figsize=(10, 6))

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
# X = XX
x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),

# print(x1_min)
# print(x1_max)
# print(x2_min)
# print(x2_max)

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

# print(xx1)
# print(xx2)

grid = np.c_[xx1.ravel(), xx2.ravel()]

# print(grid)

grid = add_ones(grid)
pro = sigmoid(np.dot(grid, theta_calc))

probs = pro.reshape(xx1.shape)

# print(probs)
plt.contour(xx1, xx2, probs, [0.5], linewidths=2, colors='black');

# Finally solution for Question No 4

print("Coefficients calculated from logistic regression")
print("\u03B80 =", theta_calc[0])
print("\u03B81 =", theta_calc[1])
print("\u03B82 =", theta_calc[2])


# QUESTION 5


# modified Logistic Regression as Perceptron Learning Algorithm

def PerceptronLearning(X, verbose=False):
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    iterations = len(X)
    h = [None] * iterations
    for i in range(iterations):
        z = np.dot(X, theta)
        #         print(z)
        for j in range(len(z)):
            h[j] = 1 if z[j] > 0 else 0

        #         print('h', h)

        gradient = np.dot(X.T, (h - y)) / y.size

        theta -= learning_rate * gradient
        #         print(theta)

        if (verbose == True and i % iterations == 0):
            #             print(f'loss: {lossfunction(h, y)} \t')
            print(theta)
    #             pass
    return theta


theta_calc_perc = PerceptronLearning(XX, False)

# Coefficients for Perceptron Learning Algorithm

print("Coefficients calculated from perceptron learning algorithm")
print("\u03B80 =", theta_calc_perc[0])
print("\u03B81 =", theta_calc_perc[1])
print("\u03B82 =", theta_calc_perc[2])


# function to preprocess input values for prediction, used by predict function
def preprocess(X):
    X.insert(0, 1)  # insert 1 at beginning
    # print(X)
    return np.array(X)


# prediction using perceptron learning algorithm
def threshold_function(z):
    return z > 0


# predicts output for given input X and learned theta

def predict(X, theta):
    preprocess(X)

    z = np.dot(X, theta)
    # print(z)

    h = threshold_function(z)
    # print(h)

    return 1 if h else 0


# question 5 run

input_X = [5.8097, 2.4711]

res = predict(input_X, theta_calc_perc)

print('Value of Y for', input_X, 'is:', res)
