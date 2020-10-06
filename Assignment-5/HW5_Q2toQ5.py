import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def phi_1(x):
    return x[1]**2-2*x[0]+3

def phi_2(x):
    return x[0]**2-2*x[1]-3

def transfrom(X):
    Z = []
    for x in X:
        z1 = phi_1(x)
        z2 = phi_2(x)
        Z.append([z1, z2])
    return np.array(Z)

if __name__=='__main__':
    # Q2
    X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    Y = np.array([-1, -1, -1, 1, 1 ,1 ,1])
    Z = transfrom(X)

    plt.plot(*zip(*[z for z, y in zip(Z, Y) if y==1]), marker='o', linestyle='none')
    plt.plot(*zip(*[z for z, y in zip(Z, Y) if y==-1]), marker='x', linestyle='none')
    plt.show()

    # Q3
    svm_poly_c = SVC(kernel='poly', degree=2, gamma=1, coef0=1, C=1e20) # inf C means hard-hardmargin
    svm_poly_c.fit(X, Y)
    # index of support vector
    sv = svm_poly_c.support_
    b = svm_poly_c.intercept_
    # dual_coef_ is only related to support vector and 
    # computed by y_n*alpha_n_, so times Y_sv to convert to alpha
    print(X[sv], b)
    print(sum(svm_poly_c.dual_coef_[0]*Y[sv]))

    # Q4
    dual_coef = svm_poly_c.dual_coef_[0]
    w = np.zeros(6)
    for sv, coef in zip(X[sv], dual_coef):
        w_n = np.array([1, 2*sv[0], 2*sv[1], 2*sv[0]*sv[1], sv[0]**2, sv[1]**2]) #(1, x1, x2, x1x2, x1^2, x2^2)
        w += coef*w_n
    w[0] += b
    print(w)