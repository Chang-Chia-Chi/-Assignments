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
            vals = line.split()
            x = [float(val) for val in vals[:-1]]
            X.append([1]+x)

            y = int(vals[-1])
            Y.append(y)

        X_data = np.array(X)
        Y_data = np.array(Y)
    return X_data, Y_data

def ridge_regression(train_X, train_Y, lamb):
    data_size = len(train_X[0])
    aug_error = train_X.T @ train_X + lamb * np.identity(data_size, dtype=float)
    inv_aug_error = np.linalg.pinv(aug_error)
    return inv_aug_error @ train_X.T @ train_Y

def valid(test_X, test_Y, w):
    u_sign = np.vectorize(sign)
    error = np.where(u_sign(test_X @ w)!=test_Y, 1, -1)
    return np.sum(error==1)

if __name__ == '__main__':
    X, Y = load_data("hw4_train.txt")
    train_X, train_Y = X[:120], Y[:120]
    val_X, val_Y = X[120:], Y[120:]

    test_X, test_Y = load_data("hw4_test.txt")

    # Q16 & Q17
    lambs = [10**(n) for n in range(-10, 3)]
    Ein = []
    Eval = []
    Eout = []
    for lamb in lambs:
        w = ridge_regression(train_X, train_Y, lamb)
        Ein.append(valid(train_X, train_Y, w))
        Eval.append(valid(val_X, val_Y, w))
        Eout.append(valid(test_X, test_Y, w))

    print(lambs)
    print(Ein)
    print(Eval)
    print(Eout)

    # Q18
    w = ridge_regression(X, Y, 1)
    print(valid(X, Y, w), valid(test_X, test_Y, w))