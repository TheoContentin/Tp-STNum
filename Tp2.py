import numpy as np
import matplotlib.pyplot as plt

theta = [5,4]

def theta1(x):
    return np.sum(x,0)/np.shape(x)[0]

def theta2(x):
    return np.median(x)

def theta3(x):
    return (np.min(x)+np.max(x))/2

Xnormal = np.random.normal(theta[0],theta[1],(500,200))

Xnormal_1 = np.zeros_like(Xnormal)
Xnormal_2 = np.zeros_like(Xnormal)
Xnormal_3 = np.zeros_like(Xnormal)

for i in range(1,len(Xnormal)):
    Xnormal_1[i] = theta1(Xnormal[:i])
    Xnormal_2[i] = theta2(Xnormal[:i])
    Xnormal_3[i] = theta3(Xnormal[:i])

fig = plt.figure()
plt.plot(Xnormal_1[:,0],label=r"$\hat{\theta}_1$")
plt.plot(Xnormal_2[:,0],label=r"$\hat{\theta}_2$")
plt.plot(Xnormal_3[:,0],label=r"$\hat{\theta}_3$")
plt.axhline(y=theta[0], color='darkgrey', linestyle='dashed')
plt.title(r"$\hat{\theta}(n)$ pour une loi normale $\mathcal{N}(5,2^2)$")
plt.xlabel("n")
plt.ylabel(r"$\hat{\theta}$")
plt.legend()
plt.grid()

bias_normal_1 = np.mean(Xnormal_1-theta[0],1)
bias_normal_2 = np.mean(Xnormal_2-theta[0],1)
bias_normal_3 = np.mean(Xnormal_3-theta[0],1)

print(bias_normal_1[499],bias_normal_2[499],bias_normal_3[499])

fig = plt.figure()

plt.plot(bias_normal_3,label=r"$\hat{\theta}_3$")
plt.plot(bias_normal_1,label=r"$\hat{\theta}_1$")
plt.plot(bias_normal_2,label=r"$\hat{\theta}_2$")

plt.axhline(y=0, color='darkgrey', linestyle='dashed')
plt.title(r"Bias moyen pour 200 echantillon de $\mathcal{N}(5,2^2)$ de taille 500")
plt.xlabel("n")
plt.ylabel(r"Biais")
plt.legend()
plt.grid()



var_normal_1 = np.var(Xnormal_1,1)
var_normal_2 = np.var(Xnormal_2,1)
var_normal_3 = np.var(Xnormal_3,1)

print(var_normal_1[499],var_normal_2[499],var_normal_3[499])

fig = plt.figure()

plt.plot(var_normal_3,label=r"$\hat{\theta}_3$")
plt.plot(var_normal_1,label=r"$\hat{\theta}_1$")
plt.plot(var_normal_2,label=r"$\hat{\theta}_2$")

plt.axhline(y=0, color='darkgrey', linestyle='dashed')
plt.title(r"Variance moyen pour 200 echantillon de $\mathcal{N}(5,2^2)$ de taille 500")
plt.xlabel("n")
plt.ylabel(r"Var")
plt.legend()
plt.grid()



#### 1.2 ####
Xuniform = np.random.uniform(0,8,500)

les_theta1,les_theta2,les_theta3=[],[],[]

for i in range(1,len(Xuniform)):
    les_theta1.append(theta1(Xuniform[:i]))
    les_theta2.append(theta2(Xuniform[:i]))
    les_theta3.append(theta3(Xuniform[:i]))

fig=plt.figure()

plt.plot(les_theta1,label=r"$\hat{\theta}_1$")
plt.plot(les_theta2,label=r"$\hat{\theta}_2$")
plt.plot(les_theta3,label=r"$\hat{\theta}_3$")
plt.title(r"$\hat{\theta}(n)$ pour une loi uniforme $\mathcal{U}(0,8)$")
plt.xlabel("n")
plt.ylabel(r"$\hat{\theta}$")
plt.legend()
plt.grid()

plt.legend()
plt.show()