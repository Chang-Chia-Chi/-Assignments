import time
from copy import deepcopy
import random
import numpy as np

# Q18
def Pocket(X, Y, iteration=50, alpha=1):
    data_size = len(X)
    vector_size = len(X[0])
    best_W = np.zeros(vector_size)
    
    # 將 Input 及 Output 向量整併成一個 numpy array
    Y_reshape = np.reshape(Y, (data_size,1))
    stack_data = np.hstack((X, Y_reshape))
    
    # 初始化計算參數
    update = 0
    W = deepcopy(best_W)
    best_count = float('inf')
    index = [i for i in range(data_size)]
    while True:
    	# index 洗牌達到 Randomized 的目的
        t = int(time.time())
        np.random.seed(t)
        rng = np.random.default_rng()
        rng.shuffle(index)

        for i in index:
            xi = stack_data[i][:vector_size]
            yi = stack_data[i][-1]
	    # 若發現錯誤則更正，並確認更新後的權重是否比之前更好
            if sign(W @ xi) != yi:
                update += 1
                W += alpha*yi*xi
                error = error_count(X,Y,W)
                if  error < best_count:
                    best_W = deepcopy(W)
                    best_count = error

            if update == iteration or best_count == 0:
                return best_W

# Q19 & Q20
def PLA(X, Y, iteration=50, alpha=1, shuffle=False):
    data_size = len(X)
    vector_size = len(X[0])
    W = np.zeros(vector_size)

    Y_reshape = np.reshape(Y, (data_size,1))
    stack_data = np.hstack((X, Y_reshape))

    count = 0
    index = [i for i in range(data_size)]

    while True:

        if shuffle == True:
            t = int(time.time())
            np.random.seed(t)
            rng = np.random.default_rng()
            rng.shuffle(index)
        
        for i in index:
            xi = stack_data[i][:vector_size]
            yi = stack_data[i][-1]

            if sign(W @ xi) != yi:
                W += alpha*yi*xi
                count += 1
            
            if count == iteration:
                return W
    

def sign(num):
    return -1 if num <= 0 else 1

def error_count(x, y, w):
    count = 0
    for xi, yi in zip(x,y):
        if sign(w @ xi) != yi:
            count += 1

    return count

if __name__ == "__main__":
    with open('train_data.txt', 'r') as file:
        X_train = []
        Y_train = []
        lines = file.readlines()
        for line in lines:
            data = line.split()
            x = [1]
            for d in data[:4]:
                x.append(float(d))
            X_train.append(x)
            Y_train.append(int(data[-1]))

    with open('test_data.txt', 'r') as file:
        X_test = []
        Y_test = []
        lines = file.readlines()
        for line in lines:
            data = line.split()
            x = [1]
            for d in data[:4]:
                x.append(float(d))
            X_test.append(x)
            Y_test.append(int(data[-1]))

    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test= np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    # Q18
    N = 100
    total_error = 0
    for _ in range(N):
        best_W = Pocket(X_train, Y_train)
        total_error += error_count(X_test, Y_test, best_W)
    error_rate = total_error/(len(X_test)*N)
    print(error_rate)

    # Q19
    total_error = 0
    for _ in range(N):
        W = PLA(X_train,Y_train, shuffle=True)
        total_error += error_count(X_test, Y_test, W)
    error_rate = total_error/(len(X_test)*N)
    print(error_rate)
    
    # Q20
    total_error = 0
    for _ in range(N):
        best_W = Pocket(X_train, Y_train, iteration=100)
        total_error += error_count(X_test, Y_test, best_W)
    error_rate = total_error/(len(X_test)*N)
    print(error_rate)
