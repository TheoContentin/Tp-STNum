import numpy as np

theta = [5,4]

def theta1(x):
    return x.sum(0)/x.size(0)

def theta2(x):
    return np.median(x)

def theta3(x):
    return (np.min(x)+np.max(x))/2

Xnormal = np.random.normal(theta[0],theta[1],500)

Xnormal_1 = np.zeros_like(Xnormal)
Xnormal_2 = np.zeros_like(Xnormal)
Xnormal_3 = np.zeros_like(Xnormal)