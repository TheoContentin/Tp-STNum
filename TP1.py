import pandas as pd # Pour lire les fichiers
import numpy as np # Pour effectuer des calculs mathématiques
import matplotlib.pyplot as plt # Pour réaliser des graphiques
from scipy import stats # Pour des calculs statistiques

tableau8 = pd.read_table("painting8.txt",
                        sep=";",
                        decimal=".",
                        header=None)

tableau8[7] = 1-tableau8.sum(1)

#Q1 : Il faut bien calculer la dernière colonne pour avoir toutes les composantes
print(tableau8.std(0))


#Q2 Voila

plt.subplots(figsize=(8, 6))

tableau8.boxplot()
plt.show()



#Q3 : Wow !

plt.subplots(figsize=(10, 10))
pd.plotting.scatter_matrix(tableau8, figsize=(15,10))
plt.show()


