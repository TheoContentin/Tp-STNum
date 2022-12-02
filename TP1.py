import pandas as pd # Pour lire les fichiers
import numpy as np # Pour effectuer des calculs mathématiques
import matplotlib.pyplot as plt # Pour réaliser des graphiques
from scipy import stats # Pour des calculs statistiques
from sklearn.preprocessing import StandardScaler # Pour normaliser les données
from sklearn import decomposition # Pour effectuer une ACP
from sklearn.cluster import AgglomerativeClustering # Pour effectuer une CAH
from sklearn.cluster import KMeans # Pour effectuer un K-means
from scipy.cluster.hierarchy import dendrogram, linkage # Pour calculer un dendrogramme

tableau8 = pd.read_table("painting8.txt",
                        sep=";",
                        decimal=".",
                        header=None)


#Q1 : Il faut bien calculer la dernière colonne pour avoir toutes les composantes
print(tableau8.std(0))


#Q2 Voila

plt.subplots(figsize=(8, 6))
tableau8.boxplot()

#Q3 : Wow !

pd.plotting.scatter_matrix(tableau8, figsize=(15,10))

#Q4 Content de calculer cette matrice

print(tableau8.corr())

#Q5 :

n = tableau8.shape[0]
p = tableau8.shape[1]
n_cp = p
norm = StandardScaler()
tableau8_acp_norm = norm.fit_transform(tableau8)
acp=decomposition.PCA(svd_solver='full', n_components=tableau8.shape[1])
coord = acp.fit_transform(tableau8_acp_norm)
val_prop = (n-1)/n * acp.explained_variance_

plt.subplots(figsize=(8, 6))

plt.bar(np.arange(1, 7+1), val_prop)
#plt.grid()
plt.title('Eboulis des valeurs propres')
plt.xlabel('Composante principale')
plt.ylabel('Valeur propre')

#Q6
part_inertie_expl = acp.explained_variance_ratio_
print(part_inertie_expl[:2].sum())

#Q7

sqrt_val_prop = np.sqrt(val_prop)

cor_var = np.zeros((p,n_cp))
for i in range(n_cp):
    cor_var[:,i] = acp.components_[i,:] * sqrt_val_prop[i]

fig, ax = plt.subplots(figsize=(10, 10))

for i in range(0, p):
    ax.arrow(0,
             0,
             cor_var[i, 0],
             cor_var[i, 1],
             head_width=0.03,
             head_length=0.03,
             length_includes_head=True)

an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Cercle de corrélations')

plt.axhline(y=0)
plt.axvline(x=0)


#q8

plt.figure()
plt.scatter(coord[:40,0],coord[:40,1], color = 'RED',label="Rembrandt")
plt.scatter(coord[40:,0],coord[40:,1], color = 'BLUE',label="Van Gogh")
plt.legend()


tableau64 = pd.read_table("painting64.txt",
                        sep=";",
                        decimal=".",
                        header=None)


print(tableau64.corr())

#Q5 :

n64 = tableau64.shape[0]
p64 = tableau64.shape[1]
n_cp64 = p64
norm = StandardScaler()
tableau64_acp_norm = norm.fit_transform(tableau64)
acp64=decomposition.PCA(svd_solver='full', n_components=tableau64.shape[1])
coord = acp64.fit_transform(tableau64_acp_norm)
val_prop64 = (n-1)/n * acp64.explained_variance_

plt.subplots(figsize=(8, 6))

plt.bar(np.arange(1, 63+1), val_prop64)
#plt.grid()
plt.title('Eboulis des valeurs propres')
plt.xlabel('Composante principale')
plt.ylabel('Valeur propre')

#Q6
part_inertie_expl64 = acp64.explained_variance_ratio_
print(part_inertie_expl64[:2].sum())

#Q7

sqrt_val_prop64 = np.sqrt(val_prop64)

cor_var64 = np.zeros((p64,n_cp64))
for i in range(n_cp):
    cor_var64[:,i] = acp64.components_[i,:] * sqrt_val_prop64[i]

fig, ax = plt.subplots(figsize=(10, 10))

for i in range(0, p64):
    ax.arrow(0,
             0,
             cor_var64[i, 0],
             cor_var64[i, 1],
             head_width=0.03,
             head_length=0.03,
             length_includes_head=True)

plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Cercle de corrélations')

plt.axhline(y=0)
plt.axvline(x=0)


#q8

plt.figure()
plt.scatter(coord[:40,0],coord[:40,1], color = 'RED',label="Rembrandt")
plt.scatter(coord[40:,0],coord[40:,1], color = 'BLUE',label="Van Gogh")
plt.legend()


acp64=decomposition.PCA(svd_solver='full', n_components=tableau64.shape[1])
coord = acp64.fit_transform(tableau64)
val_prop64 = (n-1)/n * acp64.explained_variance_

plt.subplots(figsize=(8, 6))

plt.bar(np.arange(1, 63+1), val_prop64)
#plt.grid()
plt.title('Eboulis des valeurs propres')
plt.xlabel('Composante principale')
plt.ylabel('Valeur propre')

#Q6
part_inertie_expl64 = acp64.explained_variance_ratio_
print(part_inertie_expl64[:2].sum())

#Q7

sqrt_val_prop64 = np.sqrt(val_prop64)

cor_var64 = np.zeros((p64,n_cp64))
for i in range(n_cp):
    cor_var64[:,i] = acp64.components_[i,:] * sqrt_val_prop64[i]

fig, ax = plt.subplots(figsize=(10, 10))

for i in range(0, p64):
    ax.arrow(0,
             0,
             cor_var64[i, 0],
             cor_var64[i, 1],
             head_width=0.03,
             head_length=0.03,
             length_includes_head=True)

plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Cercle de corrélations')

plt.axhline(y=0)
plt.axvline(x=0)


#q8

plt.figure()
plt.scatter(coord[:40,0],coord[:40,1], color = 'RED',label="Rembrandt")
plt.scatter(coord[40:,0],coord[40:,1], color = 'BLUE',label="Van Gogh")
plt.legend()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



cah_single = AgglomerativeClustering(distance_threshold=0,
                                     affinity='euclidean',
                                     linkage='single',
                                     n_clusters=None).fit(norm.fit_transform(tableau8))




plt.subplots(figsize=(10, 6))

plt.title("Single linkage: Normalized")
# plot the top three levels of the dendrogram
plot_dendrogram(cah_single, truncate_mode="level", p=84)



cah_single = AgglomerativeClustering(distance_threshold=0,
                                     affinity='euclidean',
                                     linkage='average',
                                     n_clusters=None).fit(norm.fit_transform(tableau8))




plt.subplots(figsize=(10, 6))

plt.title("average lainage: Normalized")
# plot the top three levels of the dendrogram
plot_dendrogram(cah_single, truncate_mode="level", p=84)


cah_single = AgglomerativeClustering(distance_threshold=0,
                                     affinity='euclidean',
                                     linkage='average',
                                     n_clusters=None).fit(tableau8)




plt.subplots(figsize=(10, 6))

plt.title("average lainage: Normalized")
# plot the top three levels of the dendrogram
plot_dendrogram(cah_single, truncate_mode="level", p=84)


cah_single = AgglomerativeClustering(distance_threshold=0,
                                     affinity='euclidean',
                                     linkage='complete',
                                     n_clusters=None).fit(norm.fit_transform(tableau8))




plt.subplots(figsize=(10, 6))

plt.title("Complete linkage: Normalized")
# plot the top three levels of the dendrogram
plot_dendrogram(cah_single, truncate_mode="level", p=84)


cah_single = AgglomerativeClustering(distance_threshold=0,
                                     affinity='euclidean',
                                     linkage='complete',
                                     n_clusters=None).fit(tableau8)




plt.subplots(figsize=(10, 6))

plt.title("complete linkage: not Normalized")
# plot the top three levels of the dendrogram
plot_dendrogram(cah_single, truncate_mode="level", p=84)





kmeans = KMeans(init='k-means++',
                max_iter=300,
                n_clusters=2,
                n_init=20).fit(tableau8)

tableau8['Classes_Kmeans'] = kmeans.labels_

tableau8.sort_values('Classes_Kmeans')['Classes_Kmeans']


n = tableau8.shape[0]
p = tableau8.shape[1]
n_cp = p
norm = StandardScaler()
tableau8_acp_norm = norm.fit_transform(tableau8)
acp=decomposition.PCA(svd_solver='full', n_components=tableau8.shape[1])
coord = acp.fit_transform(tableau8_acp_norm)

def markerchoie(m):
    s = ""
    for n in m:
        if n == 0:
            s = s+ 'v'
        if n == 1:
            s = s+ 's'

plt.figure()
for i in range(0,40):
    if tableau8['Classes_Kmeans'][i] == 0:
        plt.scatter(coord[i, 0], coord[i, 1], color='RED',
                    marker='v')
    else:
        plt.scatter(coord[i, 0], coord[i, 1], color='RED',
                    marker='s')

for i in range(40,84):
    if tableau8['Classes_Kmeans'][i] == 0:
        plt.scatter(coord[i, 0], coord[i, 1], color='BLUE',
                    marker='v')
    else:
        plt.scatter(coord[i, 0], coord[i, 1], color='BLUE',
                    marker='s')


from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='b', label='VanGogh',
                          markerfacecolor='b',linewidth=0),
Line2D([0], [0], marker='o', color='r', label='Rembrandt',
                          markerfacecolor='r',linewidth=0),
Line2D([0], [0], marker='v', color='BLACK', label='Cluster 1 ',
                          markerfacecolor='w',linewidth=0),
Line2D([0], [0], marker='s', color='BLACK', label='Cluster 2 ',
                          markerfacecolor='w',linewidth=0)]


plt.legend(handles=legend_elements)
plt.show()
