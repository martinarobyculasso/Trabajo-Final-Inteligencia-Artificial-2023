import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Cargar los arrays de NumPy
caracteristicas = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/X.npy')
etiquetas = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/y.npy', allow_pickle=True)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

def KMeans(X_train, max_iter, tol=1e-4):
    # Número de clusters
    k = 4  # Sabemos que hay 4 frutas 

    # Inicializar los centroides aleatoriamente
    np.random.seed(33)  
    centroides = X_train[np.random.choice(range(X_train.shape[0]), k, replace=False)]

    # Iterar hasta que se alcance el número máximo de iteraciones o los centroides converjan
    for i in range(max_iter):
        # Calcular la distancia entre cada punto y los centroides
        distancias = np.array([euclidean_distance(X_train, centroide) for centroide in centroides]).T

        # Asignar cada punto al cluster más cercano
        clusters = np.argmin(distancias, axis=1)

        # Guardar los centroides actuales para verificar convergencia
        nuevos_centroides = np.copy(centroides)

        # Actualizar los centroides
        for j in range(k):
            if np.any(clusters == j):
                nuevos_centroides[j] = np.mean(X_train[clusters == j], axis=0)

        # Verificar si los centroides han cambiado poco (convergencia)
        if np.all(np.abs(nuevos_centroides - centroides) < tol):
            print(f"Convergencia alcanzada en la iteración {i + 1}")
            break

        # Actualizar los centroides
        centroides = nuevos_centroides

    return clusters, centroides

# Aplicar KMeans a las características
clusters, centroides = KMeans(caracteristicas, max_iter=30)

# # Guardar los centroides en un archivo
# np.save('/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/img/img_features/centroides.npy', centroides)

# Aplicar PCA para reducir las dimensiones a 2
pca = PCA(n_components=2)
caracteristicas_pca = pca.fit_transform(caracteristicas)
centroides_pca = pca.transform(centroides)

# Definir colores para los clusters y las etiquetas originales
colores_clusters = ['magenta', 'black', 'blue', 'cyan']
colores_etiquetas = {
    'Banana': 'yellow',
    'Pera': 'green',
    'Manzana': 'red',
    'Naranja': 'orange'
}

# Crear un diccionario para mapear índices a nombres de frutas
indice_a_fruta = {0: 'Banana', 1: 'Manzana', 2: 'Naranja', 3: 'Pera'}

# Etiquetas manuales para los centroides
etiquetas_centroides = ['naranja', 'manzana', 'pera', 'banana']
# etiquetas_centroides = [2,1,3,0]

# # Guardar etiquetas en un archivo
# np.save('/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/img/img_features/etiquetas_centroides.npy', etiquetas_centroides)

# Crear una figura para los gráficos
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Graficar la división original
for label in np.unique(etiquetas):
    subset = caracteristicas_pca[etiquetas == label]
    fruta = indice_a_fruta[label]
    ax[0].scatter(subset[:, 0], subset[:, 1], 
                  label=fruta, color=colores_etiquetas[fruta], s=50, alpha=0.6, edgecolors='w')

ax[0].set_title('División Original')
ax[0].set_xlabel('Componente principal 1')
ax[0].set_ylabel('Componente principal 2')
ax[0].legend()
ax[0].grid(True)

# Graficar los resultados de KMeans
for i in range(4):
    cluster = caracteristicas_pca[clusters == i]
    ax[1].scatter(cluster[:, 0], cluster[:, 1], 
                  label=f'Cluster {i}', color=colores_clusters[i], s=50, alpha=0.6, edgecolors='w')

# Graficar los centroides con una "X" y etiquetarlos
for i, (centroide, etiqueta) in enumerate(zip(centroides_pca, etiquetas_centroides)):
    ax[1].scatter(centroide[0], centroide[1], 
                  marker='x', color='black', s=200, linewidths=3)
    ax[1].text(centroide[0], centroide[1], etiqueta, ha='right')

ax[1].set_title('Resultados de k-Means')
ax[1].set_xlabel('Componente principal 1')
ax[1].set_ylabel('Componente principal 2')
ax[1].legend()
ax[1].grid(True)

# Mostrar los gráficos
plt.show()



