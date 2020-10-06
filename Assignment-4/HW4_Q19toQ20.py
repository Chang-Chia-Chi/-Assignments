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

X, Y = load_data("hw4_train.txt")
test_X, test_Y = load_data("hw4_test.txt")

val_X = [X[:40], X[40:80], X[80:120], X[120:160], X[160:]]
val_Y = [Y[:40], Y[40:80], Y[80:120], Y[120:160], Y[160:]]

train_X = [np.delete(X,np.s_[:40], 0), 
           np.delete(X,np.s_[40:80], 0), 
           np.delete(X,np.s_[80:120], 0), 
           np.delete(X,np.s_[120:160], 0), 
           np.delete(X,np.s_[160:200], 0)]

train_Y = [np.delete(Y,np.s_[:40], 0), 
           np.delete(Y,np.s_[40:80], 0), 
           np.delete(Y,np.s_[80:120], 0), 
           np.delete(Y,np.s_[120:160], 0), 
           np.delete(Y,np.s_[160:200], 0)]

# Q19
lambs = [10**(n) for n in range(-10, 3)]
Ein = []
Eval = []
Eout = []
for lamb in lambs:
    e_in = []
    e_val = []
    e_out = []
    for i in range(5):
        w = ridge_regression(train_X[i], train_Y[i], lamb)
        e_in.append(valid(train_X[i], train_Y[i], w))
        e_val.append(valid(val_X[i], val_Y[i], w))
        e_out.append(valid(test_X, test_Y, w))
    Ein.append(e_in)
    Eval.append(e_val)
    Eout.append(e_out)

Ecv = np.sum(Eval, axis=1)
print(Ecv)

# Q20
w = ridge_regression(X, Y, 1e-8)
print(valid(X, Y, w), valid(test_X, test_Y, w))