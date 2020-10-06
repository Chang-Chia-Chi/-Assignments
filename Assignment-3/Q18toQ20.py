import random
import numpy as np

def sign(num):
    return 1 if num >= 0 else -1

def load_data(file_path):
    with open(file_path, 'r') as file:
        X = []
        Y = []
        lines = file.readlines()
        for line in lines:
            values = line.split()
            x = [float(n) for n in values[:-1]]
            X.append([1] + x)

            y = int(values[-1])
            Y.append(y)

        X_data = np.array(X)
        Y_data = np.array(Y)
    return X_data, Y_data

def sigmoid(x, y, w):
    s = -y*(w @ x)
    return 1 / (1+np.exp(-s))

def sum_grad(X, Y, w):
    N = len(X)
    grad = 0
    for i in range(N):
        grad += sigmoid(X[i], Y[i], w)*(-Y[i]*X[i])
    return grad/N

def train(train_X, train_Y, T=2000, eta=0.001):
    w = np.zeros_like(train_X[0])
    for _ in range(T):
        w -= eta * sum_grad(train_X, train_Y, w)
    return w

def stochastic_train(train_X, train_Y, T=2000, eta=0.001):
    w = np.zeros_like(train_X[0])
    for i in range(T):
        w -= eta * sigmoid(train_X[i%1000], train_Y[i%1000], w) * (-train_Y[i%1000] * train_X[i%1000])
    return w

def valid(test_X, test_Y, w):
    count = 0
    for x, y in zip(test_X, test_Y):
        if sign(w @ x) != y:
            count += 1
    return count

train_path = 'hw3_train.txt'
test_path = 'hw3_test.txt'
train_X, train_Y = load_data(train_path)
test_X, test_Y = load_data(test_path)

# Q18
w = train(train_X, train_Y)
print(valid(test_X, test_Y, w))

# Q19
w = train(train_X, train_Y, eta=0.01)
print(valid(test_X, test_Y, w))

# Q20
w = stochastic_train(train_X, train_Y)
print(valid(test_X, test_Y, w))