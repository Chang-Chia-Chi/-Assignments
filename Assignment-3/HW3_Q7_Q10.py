import math
import numpy as np

# Q7
u0 = 0.0
v0 = 0.0
learning_rate = 0.01

def Euv(u, v):
    return math.exp(u) + math.exp(2*v) + math.exp(u*v) + u**2 - 2*u*v + 2*v**2 -3*u -2*v

def dEdu(u, v):
    return math.exp(u) + v*math.exp(u*v) + 2*u -2*v -3

def dEdv(u, v):
    return 2*math.exp(2*v) + u*math.exp(u*v) -2*u + 4*v - 2

def update(u, v, iter_time, learning_rate):
    for _ in range(5):
        u -= learning_rate * dEdu(u,v)
        v -= learning_rate * dEdv(u,v)
    return u, v

u5, v5 = update(u0, v0, 5, 0.01)
print(Euv(u5, v5))

# Q10
def hassian(u, v):
    duu = math.exp(u) + v**2*math.exp(u*v) + 2
    duv = math.exp(u*v) + u*v*math.exp(u*v) -2
    dvv = 4*math.exp(2*v) + u**2*math.exp(u*v)+4

    H = np.array([[duu,duv],[duv,dvv]])
    return H

def grad(u, v):
    du = dEdu(u,v)
    dv = dEdv(u,v)
    grad = np.array([[du],[dv]])
    return grad

def update_Q10(vector, iter_time):
    for _ in range(iter_time):
        H = hassian(vector[0][0], vector[1][0])
        inv_H = np.linalg.inv(H)
        g = grad(vector[0][0], vector[1][0])
        vector -= inv_H @ g
    return vector

vector = update_Q10(np.array([[u0],[v0]]), 5)
print(Euv(vector[0][0], vector[1][0]))