import random
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        X = []
        Y = []
        for line in lines:
            d, i, s = line.split()
            X.append([1, float(i), float(s)])
            Y.append(int(float(d)))
    return X, Y

def binary_Y(Y, n):
    b_Y = []
    for y in Y:
        if y == n:
            b_Y.append(1)
        else:
            b_Y.append(-1)
    return b_Y

def get_weight(coef, sv_index, X):
    X_array = np.array(X)
    X_sv = np.array(X_array[sv_index])
    w = np.zeros_like(X_sv[0])
    for c, x in zip(coef, X_sv):
        w += c * x
    return w

def valid(svc, test_X, test_Y):
    predict_Y = svc.predict(test_X)
    error = sum(predict_Y != test_Y)
    return error

if __name__=='__main__':
    # Q15
    train_X, train_Y = load_data('train.txt')
    Y_0 = binary_Y(train_Y, 0)
    svm_lin_c = SVC(kernel='linear', C=0.01)
    svm_lin_c.fit(train_X, Y_0)

    sv_index = svm_lin_c.support_
    coef = svm_lin_c.dual_coef_[0]
    w = get_weight(coef, sv_index, train_X)
    print(w)

    # Q16 & Q17
    errors = []
    alpha = []
    problems = [0, 2, 4, 6, 8]
    for p in problems:
        Y_p = binary_Y(train_Y, p)
        svm_poly_c = SVC(kernel='poly', degree=2, gamma=1, coef0=1, C=0.01)
        svm_poly_c.fit(train_X, Y_p)

        sv_index = svm_poly_c.support_
        Y_sv = np.array(Y_p)[sv_index]
        coef = svm_poly_c.dual_coef_[0]

        alpha.append(sum(coef*Y_sv))
        errors.append(valid(svm_poly_c, train_X, Y_p))

    print(errors, alpha)

    # Q18
    test_X, test_Y = load_data('test.txt')

    Cs = [0.001, 0.01, 0.1, 1, 10]
    w = []
    errors = []
    svs = []
    t_Y = binary_Y(test_Y, 0)   
    for c in Cs:
        svm_gauss_c = SVC(kernel='rbf', gamma=100, C=c)
        svm_gauss_c.fit(train_X, Y_0)

        sv_index = svm_gauss_c.support_
        coef = svm_gauss_c.dual_coef_[0]

        w.append(np.linalg.norm(get_weight(coef, sv_index, train_X))) 
        errors.append(valid(svm_gauss_c, test_X, t_Y))
        svs.append(len(sv_index))

    print(w, errors, svs)

    # Q19
    errors = []
    gamma = [1, 10, 100, 1000, 10000]
    for g in gamma:
        svm_gauss_c = SVC(kernel='rbf', gamma=g, C=0.1)
        svm_gauss_c.fit(train_X, Y_0)
        errors.append(valid(svm_gauss_c, test_X, t_Y))

    print(errors)

    # Q20
    count = [0,0,0,0,0]
    for _ in range(100):
        errors = []
        for g in gamma:
            X_train, X_valid, y_train, y_valid = train_test_split(
                    train_X, train_Y, test_size=1000/7291, random_state=42)
            Y_0 = binary_Y(y_train, 0)

            svm_gauss_c = SVC(kernel='rbf', gamma=g, C=0.1)
            svm_gauss_c.fit(X_train, Y_0)

            sv_index = svm_gauss_c.support_
            coef = svm_gauss_c.dual_coef_[0]

            y_val = binary_Y(y_valid, 0)
            errors.append(valid(svm_gauss_c, X_valid, y_val))
        
        errors = np.array(errors)
        index = np.argmin(errors)
        count[index] += 1

    print(count)