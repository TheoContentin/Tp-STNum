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
plt.savefig("dist_speed.jpg", format="jpg")

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
plt.savefig("Reg_lin_simple_moustache.jpg", format="jpg")

plt.subplots()
plt.hist(cars['residu'], density=True)
plt.xlabel('Résidu')
plt.ylabel('Histogramme')
plt.title('Régression linéaire simple')
plt.savefig("Reg_lin_simple_hist.jpg", format="jpg")

## Q3 ##
plt.subplots()
plt.scatter(cars['speed'], cars['dist'])
plt.plot(cars['speed'], reglin_sim.fittedvalues, color='red', label='Droite de régression')
plt.xlabel('vitesse')
plt.ylabel('distance')
plt.title('Régression linéaire simple')
plt.grid()
plt.legend()
plt.savefig("Reg_lin_simple.jpg", format="jpg")

cars['dist_ajust'] = reglin_sim.fittedvalues
bissec = np.linspace(cars['dist'].min(),cars['dist'].max(),5)

plt.subplots()
plt.scatter(cars['dist'], cars['dist_ajust'])
plt.plot(bissec, bissec, linestyle='dashed', lw=2, color='blue')
plt.grid()
plt.xlabel('distance')
plt.ylabel('distance ajustée')
plt.title('Régression linéaire simple')
plt.savefig("Reg_lin_simple_ajust.jpg", format="jpg")
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
reglin_multi_model = ols('Life_Exp ~ Population + Income + Illiteracy + Murder + HS_Grad + Frost + Area', data = state)
reglin_multi = reglin_multi_model.fit()
print(reglin_multi.summary())

# Area est la moins significative : on l'a retire
reglin_multi_model = ols('Life_Exp ~ Population + Income + Illiteracy + Murder + HS_Grad + Frost', data = state)
reglin_multi = reglin_multi_model.fit()
print(reglin_multi.summary())

# Illiteracy est la moins significative : on l'a retire
reglin_multi_model = ols('Life_Exp ~ Population + Income + Murder + HS_Grad + Frost', data = state)
reglin_multi = reglin_multi_model.fit()
print(reglin_multi.summary())

# Income est la moins significative : on l'a retire
reglin_multi_model = ols('Life_Exp ~ Population + Murder + HS_Grad + Frost', data = state)
reglin_multi = reglin_multi_model.fit()
print(reglin_multi.summary())

# # On peut peut-être retirer Population qui reste au-dessus de 0.05
# reglin_multi_model = ols('Life_Exp ~ Murder + HS_Grad + Frost', data = state)
# reglin_multi = reglin_multi_model.fit()
# print(reglin_multi.summary())

# On ne retire plus rien

## Q7 ##
state['Life_Exp_ajust'] = reglin_multi.fittedvalues
bissec = np.linspace(state['Life_Exp'].min(),state['Life_Exp'].max(),5)

plt.subplots()
plt.scatter(state['Life_Exp'], state['Life_Exp_ajust'])
plt.plot(bissec, bissec, linestyle='dashed', lw=2, color='blue')
plt.grid()
plt.xlabel('Life_Exp')
plt.ylabel('Life_Exp ajusté')
plt.savefig("Reg_lin_multi_ajust.jpg", format="jpg")

state['residu'] = reglin_multi.resid

moy_res = state.residu.mean()
print(f'Moyenne des résidus : {moy_res:.2f}')

var_res = state.residu.var()
print(f'Variance des résidus : {var_res:.2f}')

plt.subplots()
plt.boxplot(state['residu'], labels=['Boîte à moustaches'])
plt.grid()
plt.ylabel('Résidu')
plt.title('Régression linéaire multiple')
plt.savefig("Reg_lin_multi_moustache.jpg", format="jpg")

plt.subplots()
plt.hist(state['residu'], density=True)
plt.grid()
plt.xlabel('Résidu')
plt.ylabel('Histogramme')
plt.title('Régression linéaire multiple')
plt.savefig("Reg_lin_multi_hist.jpg", format="jpg")
# plt.show()

# Prévision
a_prevoir = pd.DataFrame({'Population': [30000], 'Murder': [0], 'HS_Grad': [80], 'Frost': [0]})
prev = reglin_multi.predict(a_prevoir)
print(f'Esperence de vie prevue pour Murder=0, HS_Grad=80 et Frost=0 : {prev[0]:.2f} ans')

## Q8 ##
from statsmodels.stats.outliers_influence import variance_inflation_factor # Pour les VIF
from statsmodels.tools.tools import add_constant # Pour ajouter la constante (calcul de VIF)

X = add_constant(state[['Population', 'Murder', 'HS_Grad', 'Frost']])

VIF = pd.DataFrame()
VIF['Variables'] = X.columns
VIF['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(VIF[1:len(X.columns)]) # pas de problèmes de colinéarité : tous les VIF sont inférieurs à 10

## Q10 ##
from sklearn.linear_model import LinearRegression # Pour la régression linéaire
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Pour les critères d'erreur

X = state[['Murder', 'HS_Grad', 'Frost']]
y = state['Life_Exp']

apprent = np.random.binomial(n=1, p=0.2, size=state.shape[0])

X_train = X[apprent == 0]
y_train = y[apprent == 0]

X_test = X[apprent == 1]
y_test = y[apprent == 1]

reg = LinearRegression().fit(X_train, y_train)

RMSE = np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))
print(f'RMSE : {RMSE:.2f}')

MAPE = mean_absolute_percentage_error(y_test, reg.predict(X_test)) * 100
print(f'MAPE (%): {MAPE:.2f}')

## Q11 ##
apprent2 = np.random.binomial(n=1, p=0.2, size=state.shape[0])

X_train2 = X[apprent2 == 0]
y_train2 = y[apprent2 == 0]

X_test2 = X[apprent2 == 1]
y_test2 = y[apprent2 == 1]

reg2 = LinearRegression().fit(X_train2, y_train2)

RMSE2 = np.sqrt(mean_squared_error(y_test2, reg2.predict(X_test2)))
print(f'RMSE autre échantillon: {RMSE2:.2f}')

MAPE2 = mean_absolute_percentage_error(y_test2, reg2.predict(X_test2)) * 100
print(f'MAPE autre échantillon (%): {MAPE2:.2f}')

## Q12 ##
K = 5

bloc = np.random.randint(low=0, high=K, size=state.shape[0])

nb = np.empty(K, dtype=int)
RMSE = np.empty(K, dtype=float)
MAPE = np.empty(K, dtype=float)

for i in range(K):
    nb[i] = X[bloc == i].shape[0]

    X_train = X[bloc != i]
    y_train = y[bloc != i]

    X_test = X[bloc == i]
    y_test = y[bloc == i]

    reg = LinearRegression().fit(X_train, y_train)

    RMSE[i] = np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))
    MAPE[i] = mean_absolute_percentage_error(y_test, reg.predict(X_test)) * 100

RMSE_CV = np.sum(nb * RMSE / np.sum(nb))
print(f'RMSE CV : {RMSE_CV:.2f}')

MAPE_CV = np.sum(nb * MAPE / np.sum(nb))
print(f'MAPE CV (%) : {MAPE_CV:.2f}')

plt.subplots()
plt.bar(range(K), RMSE)
plt.axhline(y=RMSE_CV, label='RMSE CV', color='red')
plt.xlabel('Bloc')
plt.ylabel('RMSE')
plt.title('Validation croisée')
plt.legend(loc='best')
plt.savefig("Validation_croisee_RMSE.jpg", format="jpg")

plt.subplots()
plt.bar(range(K), MAPE)
plt.axhline(y=MAPE_CV, label='MAPE CV', color='red')
plt.xlabel('Bloc')
plt.ylabel('MAPE')
plt.title('Validation croisée')
plt.legend(loc='best')
plt.savefig("Validation_croisee_MAPE.jpg", format="jpg")
# plt.show()
