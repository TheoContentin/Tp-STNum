import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Pour lire les fichiers

#### 1 ####
cars = pd.read_table("cars.txt",
                        sep=" ",
                        decimal=".")

## Q1 ##
dist_speed=plt.figure()
plt.plot(cars['speed'],cars['dist'],'.')
plt.xlabel('vitesse')
plt.ylabel('distance')
plt.title('distance=f(vitessse)')
plt.grid()

## Q2 ##
from statsmodels.formula.api import ols # Pour la régression linéaire

reglin_sim_model = ols('dist ~ speed', data = cars)
reglin_sim = reglin_sim_model.fit()
print(reglin_sim.summary())

# Analyse des résidus #
cars['residu'] = reglin_sim.resid

moy_res = cars.residu.mean()
print(f'Moyenne des résidus : {moy_res:.2f}')

var_res = cars.residu.var()
print(f'Variance des résidus : {var_res:.2f}')

plt.subplots()
plt.boxplot(cars['residu'], labels=['Boîte à moustaches'])
plt.grid()
plt.ylabel('Résidu')
plt.title('Régression linéaire simple')

plt.subplots()
plt.hist(cars['residu'], density=True)
plt.xlabel('Résidu')
plt.ylabel('Histogramme')
plt.title('Régression linéaire simple')

## Q3 ##
plt.subplots(figsize=(8, 6))
plt.scatter(cars['speed'], cars['dist'])
plt.plot(cars['speed'], reglin_sim.fittedvalues, color='red', label='Droite de régression')
plt.xlabel('vitesse')
plt.ylabel('distance')
plt.title('Régression linéaire simple')
plt.grid()
plt.legend()
# plt.show()

## Q4 ##
alpha=-17.5791
beta=3.9324
def prevision(vit):
    return alpha+beta*vit

print('prévision de la distance d arrêt pour une vitesse de O mph : '+str(prevision(0))+' m \n')
print('prévision de la distance d arrêt pour une vitesse de 40 mph : '+str(prevision(40))+' m \n')

a_prevoir = pd.DataFrame({'speed': [40]})
prev = reglin_sim.predict(a_prevoir)
print(f'Valeur prévue : {prev[0]:.2f}')

#### 2 ####
state = pd.read_table("state.txt",
                        sep=" ",
                        decimal=".")

## Q5 ##
