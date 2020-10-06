import random
import numpy as np
import matplotlib.pyplot as plt

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

def linear_reg(data_set, label):
    size = len(data_set)
    data_set_temp = []
    for i in range(size):
        data_set_temp.append(np.append([1], data_set[i]))

    X = np.array(data_set_temp)
    Y = np.array(label)
    pinv_X = np.linalg.pinv(X)    
    return pinv_X @ Y

def valid(data_set, label):
    w = linear_reg(data_set, label)
    size = len(data_set)
    count = 0
    for i in range(size):
        if sign(w @ data_set[i]) != label[i]:
            count += 1
    
    return count
        
# visualize
data_set, label = data_generator()
inside = [d for d, l in zip(data_set, label) if l==1]
outside = [d for d, l in zip(data_set, label) if l==-1]
plt.plot(*zip(*inside), marker='.', linestyle='none')
plt.plot(*zip(*outside), marker='.', linestyle='none')
w = linear_reg(data_set, label)
slop = w[1]/w[2]
x = [-1, 1]
y = [-p*slop-w[0] for p in x]
plt.plot(x, y)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

Ein = 0
time = 1000
for _ in range(time):
    Ein += valid(data_set, label)
print(Ein/time)
