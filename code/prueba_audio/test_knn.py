import os
import librosa
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import warnings

# Ignorar warnings específicos
warnings.filterwarnings("ignore", message="n_fft=2048 is too large for input signal of length=")


# Funciones -------------------------------------------------------------------------------------------

# Función para extraer las características de un archivo de audio
def extract_features(path):
    # Cargar el audio
    y, sr = librosa.load(path, sr=None)

    features = []

    n_samples = len(y)
    max_zcr_total = np.max(librosa.feature.zero_crossing_rate(y=y))
    
    # Dividir el audio en 10 partes iguales
    segment_length = len(y) // 10
    segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(10)]
    
    i = 0
    for segment in segments:
        i += 1

        if i == 2:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
            mean_mfcc_5_s2 = np.std(mfcc[4])

        elif i == 3:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
            max_mfcc_2_s3 = np.max(mfcc[1])

            mean_amp_s3 = np.mean(np.abs(segment))
        
        elif i == 4:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
            mean_mfcc_1_s4 = np.mean(mfcc[0])

            mean_amp_s4 = np.mean(np.abs(segment))

            max_amp_s4 = np.max(np.abs(segment))
        
        elif i == 5:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
            max_mfcc_4_s5 = np.max(mfcc[3])
            mean_mfcc_5_s5 = np.mean(mfcc[4])
        
        elif i == 7:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
            mean_mfcc_1_s7 = np.mean(mfcc[0])

            max_amp_s7 = np.max(np.abs(segment))
        
        elif i == 8:
            zcr = librosa.feature.zero_crossing_rate(y=segment)
            mean_zcr_s8 = np.mean(zcr)

    features = [max_zcr_total, mean_mfcc_1_s4, mean_amp_s4, max_amp_s4, max_mfcc_4_s5, n_samples, max_mfcc_2_s3, mean_mfcc_1_s7, mean_mfcc_5_s5, mean_mfcc_5_s2, mean_zcr_s8, max_amp_s7, mean_amp_s3]

    return features

# Función para escalar las características extraídas
def scale_features(features, max_vals, min_vals):
    scaled_features = (features - min_vals) / (max_vals - min_vals + 1e-10)
    return scaled_features

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# K-Nearest Neighbors
def KNN(X_train, y_train, X_new, k):
    # Se calculan las distancias entre el punto a clasificar y los puntos de entrenamiento
    distancias = []
    for i in range(len(X_train)):
        distancias.append(euclidean_distance(X_train[i], X_new))
    distancias = np.array(distancias)

    # Se ordenan las distancias de menor a mayor
    distancias_sorted_indices = np.argsort(distancias)

    # Se obtienen los índices de los k vecinos más cercanos
    k_indices = distancias_sorted_indices[:k]

    # Se obtienen las etiquetas de los k vecinos más cercanos
    k_etiquetas = y_train[k_indices]

    # Se obtiene la etiqueta más común
    etiqueta = np.bincount(k_etiquetas).argmax()

    return etiqueta

# ---------------------------------------------------------------------------------------------------------

# Características, etiquetas y max/min valores de características seleccionadas del dataset de entrenamiento
# Se cargan los archivos .npy almacenados
X_dataset = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/audio_features/test/X.npy')
y_dataset = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/audio_features/test/y.npy')
max_dataset_values = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/audio_features/test/max_values.npy')
min_dataset_values = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/audio_features/test/min_values.npy')

# Dataset de prueba (test)
test_processed_path = '/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/test_procesados'

# Etiquetas
labels = ['banana', 'manzana', 'naranja', 'pera']

# Procesamiento y extracción de características de los audios de prueba ------------------------------------

# Extracción de características de los audios de prueba
X_test = []
y_test = []
file_names = []  # Lista para almacenar los nombres de los archivos

for label_index, label in enumerate(labels):
    folder = os.path.join(test_processed_path, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if file_path.endswith(('.wav', '.mp3', '.flac')):
            feats = extract_features(file_path)
            X_test.append(feats)
            y_test.append(label_index)
            file_names.append(file)  # Almacenar el nombre del archivo

X_test = np.array(X_test)
y_test = np.array(y_test)

# Escalado
X_test = scale_features(X_test, max_dataset_values, min_dataset_values)

# Algoritmo de clasificación ---------------------------------------------------------------------------------

k = 7  # Número de vecinos más cercanos
y_pred = []

for i in range(len(X_test)):
    y_pred.append(KNN(X_dataset, y_dataset, X_test[i], k))

y_pred = np.array(y_pred)
# print(y_pred)

# Resultados --------------------------------------------------------------------------------------------------

confusion_matrix = np.zeros((len(labels), len(labels)))

for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1

# Imprimir los nombres de los audios clasificados incorrectamente
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f'Archivo clasificado incorrectamente: {file_names[i]}')

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta verdadera')
plt.title('Matriz de confusión - Audios')
plt.show()

# Realizar PCA para reducir a 3 dimensiones
pca = PCA(n_components=3)
X_pca_train = pca.fit_transform(X_dataset)
X_pca_test = pca.transform(X_test)

# Crear una figura 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colores para cada etiqueta
colors = {'manzana': 'red', 'naranja': 'orange', 'pera': 'green', 'banana': 'yellow'}
label_colors = [colors[label] for label in labels]

# Plotear los datos de entrenamiento
for label_index, label in enumerate(labels):
    indices = np.where(y_dataset == label_index)
    ax.scatter(X_pca_train[indices, 0], X_pca_train[indices, 1], X_pca_train[indices, 2], c=label_colors[label_index], label=label)

# Plotear los datos de prueba con cruces
for i in range(len(X_pca_test)):
    color = 'purple' if y_test[i] != y_pred[i] else 'black'
    ax.scatter(X_pca_test[i, 0], X_pca_test[i, 1], X_pca_test[i, 2], c=color, marker='*', s=150)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3D de las características del dataset de entrenamiento y prueba')
ax.legend()
plt.show()

# Calcular la precisión
accuracy = np.sum(y_test == y_pred) / len(y_test) * 100
print(f'Precisión: {accuracy:.2f}%')

# ---------------------------------------------------------------------------------------------------------