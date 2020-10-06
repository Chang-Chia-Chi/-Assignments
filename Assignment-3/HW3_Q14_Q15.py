import random
import numpy as np

def cir_sign(x, y, radius):
    return 1 if x**2+y**2-radius >= 0 else -1

def sign(num):
    return 1 if num >= 0 else -1

def data_generator(N=1000, noise=0.1, radius=0.6):
    data_set = []

    for _ in range(N):
        xi = random.uniform(-1, 1)
        yi = random.uniform(-1, 1)
        data_set.append([xi,yi])

    labels = []
    prob = [1-noise, noise]
    flag = [1, -1]
    for data in data_set:
        s = cir_sign(data[0], data[1], radius)*random.choices(flag, weights=prob)[0]
        labels.append(s)
    
    return np.array(data_set), np.array(labels)

def transform(data_set):
    trans_data = []
    for data in data_set:
        x, y = data[0], data[1]
        xy, x2, y2 = x*y, x**2, y**2
        trans_data.append([1,x,y,xy,x2,y2])
    
    return np.array(trans_data)

def linear_reg(trans_data, label):
    pinv_X = np.linalg.pinv(trans_data)
    return pinv_X @ label

def valid(data_set, label, w):
    size = len(data_set)
    count = 0
    for i in range(size):
        if sign(w @ data_set[i]) != label[i]:
            count += 1
    
    return count

# Q14
data_set, label = data_generator()
trans_data = transform(data_set)
W = []
W.append(np.array([-1, -0.05, 0.08, 0.13, 1.5, 1.5]))
W.append(np.array([-1, -0.05, 0.08, 0.13, 1.5, 15]))
W.append(np.array([-1, -0.05, 0.08, 0.13, 15, 1.5]))
W.append(np.array([-1, -1.5, 0.08, 0.13, 0.05, 1.5]))
W.append(np.array([-1, -1.5, 0.08, 0.13, 0.05, 0.05]))
error = [0] * 5
for i in range(5):
    w = W[i]
    count = 0
    for data, l in zip(trans_data, label):
        if sign(w @ data) != l:
            count += 1
    error[i] = count
print(error)

# Q15
Eout = 0
time = 1000
for _ in range(time):
    data_set, label = data_generator()
    trans_data = transform(data_set)
    w = linear_reg(trans_data, label)

    test_data_set, test_label = data_generator()
    trans_test = transform(test_data_set)
    Eout += valid(trans_test, test_label, w)
print(Eout/time)