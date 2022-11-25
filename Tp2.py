import numpy as np
import matplotlib.pyplot as plt

theta = [5,4]

def theta1(x):
    return np.sum(x,0)/np.size(x)

def theta2(x):
    return np.median(x)

def theta3(x):
    return (np.min(x)+np.max(x))/2

Xnormal = np.random.normal(theta[0],theta[1],500)

Xnormal_1 = np.zeros_like(Xnormal)
Xnormal_2 = np.zeros_like(Xnormal)
Xnormal_3 = np.zeros_like(Xnormal)

for i in range(1,len(Xnormal)):
    Xnormal_1[i] = theta1(Xnormal[:i])
    Xnormal_2[i] = theta2(Xnormal[:i])
    Xnormal_3[i] = theta3(Xnormal[:i])

fig = plt.figure()
plt.plot(Xnormal_1,label=r"$\hat{\theta}_1$")
plt.plot(Xnormal_2,label=r"$\hat{\theta}_2$")
plt.plot(Xnormal_3,label=r"$\hat{\theta}_3$")
plt.title(r"$\hat{\theta}(n)$ pour une loi normale $\mathcal{N}(5,2^2)$")
plt.xlabel("n")
plt.ylabel(r"$\hat{\theta}$")
plt.legend()
plt.grid()
plt.show()