import time
import random
import numpy as np

t = int(time.time())
np.random.seed(t)

A = ('1g', '2o', '3g', '4o', '5g', '6o')
B = ('1o', '2g', '3o', '4g', '5o', '6g')
C = ('1o', '2o', '3o', '4g', '5g', '6g')
D = ('1g', '2g', '3g', '4o', '5o', '6o')

bag = []
for _ in range(1000):
    bag.append(A)
    bag.append(B)
    bag.append(C)
    bag.append(D)

# Q13
N = 10000
count = 0
for _ in range(N):
    sample = random.sample(bag, 5)
    if all(item[0] == '1o' for item in sample):
        count += 1

print(count/N)

# Q14
count = 0
for _ in range(N):
    sample = random.sample(bag, 5)
    for nums in zip(*sample):
        if all(num[1] == 'o' for num in nums):
            count += 1
            break

print(count/N)        