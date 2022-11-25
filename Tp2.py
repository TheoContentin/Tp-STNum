import matplotlib.pyplot as plt
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




































#### 1.2 ####
Xuniform = np.random.uniform(0,8,500)

les_theta1,les_theta2,les_theta3=[],[],[]

for i in range(len(Xuniform)):
    les_theta1.append(theta1(Xuniform[:i]))
    les_theta2.append(theta2(Xuniform[:i]))
    les_theta3.append(theta3(Xuniform[:i]))

fig=plt.figure()

plt.plot(les_theta1,Xuniform,label='theta1')
plt.plot(les_theta1,Xuniform,label='theta2')
plt.plot(les_theta1,Xuniform,label='theta3')

plt.legend()
plt.show()