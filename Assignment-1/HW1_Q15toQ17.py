import time
import random
import numpy as np

def PLA(X, Y, alpha=1, shuffle=False):
    data_size = len(X)
    vector_size = len(X[0])
    W = np.zeros(vector_size)
    
    Y = np.reshape(Y, (data_size,1))
    stack_data = np.hstack((X,Y))

    if shuffle == True:
        t = int(time.time())
        np.random.seed(t)
        rng = np.random.default_rng()
        rng.shuffle(stack_data)

    # 題意為計算權重需"更新"幾次
    # count 計算總更新次數
    # cycle 計算每次迴圈更新幾次，若某次迴圈無更新，代表迭代完成
    count = 0
    while True:
        cycle = 0
        for i in range(data_size):
            xi = stack_data[i][:vector_size]
            yi = stack_data[i][-1]

            if sign(W @ xi) != yi:
                W += alpha*yi*xi
                count += 1
                cycle += 1
        
        if cycle == 0:
            break

    return W, count

def sign(num):
    return -1 if num <= 0 else 1

def test(x, y, w):
    count = 0
    vector_size = len(x[0])
    for xi, yi in zip(x,y):
        total = w @ xi

        if sign(total) != yi:
            count += 1

    return count

if __name__ == "__main__":
    # read data from file
    with open('data.txt', 'r') as file:
        X = []
        Y = []
        lines = file.readlines()
        for line in lines:
            data = line.split()
            x = [1]
            for d in data[:4]:
                x.append(float(d))
            X.append(x)
            Y.append(int(data[-1]))

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Q15
    print(PLA(X,Y, shuffle=False))

    # Q16
    total = 0
    for _ in range(2000):
        _, count = PLA(X,Y, shuffle=True)
        total += count
    print(total/2000)
    
    # Q17
    total = 0
    for _ in range(2000):
        _, count = PLA(X,Y, alpha=0.5, shuffle=True)
        total += count
    print(total/2000)

    W, _ = PLA(X,Y, alpha=0.5, shuffle=True)

    print(test(X,Y,W))