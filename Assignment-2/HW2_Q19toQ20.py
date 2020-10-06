import time
import random
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as file:
        X = []
        Y = []
        lines = file.readlines()
        for line in lines:
            values = line.split()
            x = [float(n) for n in values[:-1]]
            X.append(x)

            y = int(values[-1])
            Y.append(y)

        X_data = np.transpose(np.asarray(X))
        Y_data = np.asarray(Y)
    return X_data, Y_data

def sign(num, s=1, theta=0):
    return -1*s if num <= theta else 1*s

def multi_decision_stump(X, Y):
    dim = len(X[0])

    best_s = 1
    best_theta = 0
    min_error = float('inf')
    for x in X:
    	# 將 X 及 Y 排序 (利用argsort，再產生對應的資料)
        sort_i = np.argsort(x)
        sort_x = x[sort_i]
        sort_y = Y[sort_i]
        thetas = np.array([float('-inf')]+[(sort_x[i]+sort_x[i+1])/2 for i in range(dim-1)]+[float('inf')])

        s = 1
        good_theta = 0
        error = float('inf')
        for theta in thetas:
            # positive 為 s = 1 的情況 / negative 為 s = -1 的情況
            y_positive = np.where(sort_x > theta, 1, -1)
            y_negative = np.where(sort_x < theta, 1, -1)
	    
	    # 與原資料比較，計算錯誤數量
            error_positive = sum(y_positive!=sort_y)
            error_negative = sum(y_negative!=sort_y)

	    # 依錯誤數量更新參數
            if error_positive < error:
                s = 1
                error = error_positive
                good_theta = theta
            
            if error_negative < error:
                s = -1
                error = error_negative
                good_theta = theta
                
	# 確認每個特徵 Xn 的訓練效果，並紀錄最佳值
        if error < min_error:
            best_s = s
            best_theta = good_theta
            min_error = error
            
    return min_error/dim, best_theta, best_s

def valid(X, Y, theta, s):
    dim = len(X[0])
    E_out = []
    u_sign = np.vectorize(sign)
    for x in X:
        Y_test = u_sign(x, theta=theta, s=s)
        E_out.append(sum(Y_test!=Y))
    
    return min(E_out)/dim

if __name__ == "__main__":
    train_X, train_Y = load_data("hw2_train.txt")
    test_X, test_Y = load_data("hw2_test.txt")
    E_in, theta, s = multi_decision_stump(train_X, train_Y)
    E_out = valid(test_X, test_Y, theta, s)
    print(E_in, E_out, theta, s)
