import time
import random
import numpy as np

def Y_generator(X, noise=0.2):
    weights = [1-noise, noise]
    noise_flag = ["Not_effected", "effected"]

    size = (X.shape)[0]
    Y = np.empty(size)
    for i in range(size):
        flag = random.choices(noise_flag, weights=weights)
        if flag[0] == "Not_effected":
            Y[i] = sign(X[i])
        elif flag[0] == "effected":
            Y[i] = -sign(X[i])
    
    return Y


def sign(num, s=1, theta=0):
    return -1*s if num <= theta else 1*s

def decision_stump(X, Y, lower_bound=-1, upper_bound=1):
    # 將 X 及 Y 排序 (利用argsort，再產生對應的資料)
    size = (X.shape)[0]
    sort_i = np.argsort(X)
    sort_X = X[sort_i]
    sort_Y = Y[sort_i]
    
    # 初始化，theta 利用相鄰資料的平均值計算
    s = 1
    best_theta = 0
    min_error = float('inf')
    thetas = np.array([lower_bound] + [(X[i]+X[i+1])/2 for i in range(size-1)] + [upper_bound])
    for theta in thetas:
        # positive 為 s = 1 的情況 / negative 為 s = -1 的情況
        y_positive = np.where(sort_X > theta, 1, -1)
        y_negative = np.where(sort_X < theta, 1, -1)

        # 與原資料比較，計算錯誤數量
        error_positive = sum(y_positive!=sort_Y)
        error_negative = sum(y_negative!=sort_Y)

        # 依錯誤數量更新參數
        if error_positive < min_error:
            min_error = error_positive
            best_theta = theta
            s = 1
        
        if error_negative < min_error:
            min_error = error_negative
            best_theta = theta
            s = -1

    return min_error, best_theta, s

# Q16
"""
h = s*sign(x-theta)，theta 將 [-1, 1] 分成可能不等的兩塊，而無 noise 的 f(x) 即為 theta = 0, s = 1 的狀況。
因此，可以利用座標偏心的想法出發，偏心 theta 後，不考慮 noise 的情況下 正確的部份剩下 1- theta/2， 錯誤的部份為 theta / 2

s = 1，偏心後正確機率為 theta/2*0.2 + (1-theta/2)*0.8 ; s = -1 ，偏心後正確機率為 theta/2*0.8 + (1-theta/2)*0.2
經整理後可得到答案為 0.5 + s*0.3*(abs(theta)-1) (因為左右對稱，所以 theta 無論正負結果相同)
"""

# Q17 & Q18
if __name__ == "__main__":
    N = 5000
    SIZE = 20
    error_in = 0
    error_out = 0
    for i in range(N):
        t = int(time.time())
        np.random.seed(t)

        X = np.random.uniform(-1.0, 1.0, SIZE)
        Y = Y_generator(X)

        error, theta, s = decision_stump(X, Y)
        error_in += error
        error_out += SIZE*(0.5 + 0.3*s*(abs(theta) - 1))

    E_in = error_in/(N*SIZE)
    E_out = error_out/(N*SIZE)
    print(E_in, E_out)
    

