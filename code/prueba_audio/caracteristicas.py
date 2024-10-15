import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# Ignorar warnings específicos
warnings.filterwarnings("ignore", message="n_fft=2048 is too large for input signal of length=")

# ------------------------------------ Funciones ------------------------------------

def extract_features(path):
    # Cargar el audio
    y, sr = librosa.load(path, sr=None)

    features = []
    feature_labels = []

    n_samples = len(y)
    feature_labels.append('n_samples')

    max_zcr_total = np.max(librosa.feature.zero_crossing_rate(y=y))
    feature_labels.append('max_zcr_total')
    
    # Dividir el audio en 10 partes iguales
    segment_length = len(y) // 10
    segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(10)]
    
    mean_zcr_segment = []
    mean_amp_segment = []
    max_amp_segment = []
    mean_mfcc_1_segment = []
    mean_mfcc_2_segment = []
    mean_mfcc_5_segment = []
    max_mfcc_2_segment = []
    max_mfcc_4_segment = []
    max_mfcc_5_segment = []
    std_mfcc_2_segment = []
    std_mfcc_4_segment = []
    std_mfcc_5_segment = []
    mean_spectral_centroid_segment = []
    mean_spectral_bandwidth_segment = []

    lbl_mean_zcr_segment = []
    lbl_mean_amp_segment = []
    lbl_max_amp_segment = []
    lbl_mean_mfcc_1_segment = []
    lbl_mean_mfcc_2_segment = []
    lbl_mean_mfcc_5_segment = []
    lbl_max_mfcc_2_segment = []
    lbl_max_mfcc_4_segment = []
    lbl_max_mfcc_5_segment = []
    lbl_std_mfcc_2_segment = []
    lbl_std_mfcc_4_segment = []
    lbl_std_mfcc_5_segment = []
    lbl_mean_spectral_centroid_segment = []
    lbl_mean_spectral_bandwidth_segment = []

    
    i = 0
    for segment in segments:
        i += 1

        # Calcular el "mean zcr" de cada segmento
        zcr = librosa.feature.zero_crossing_rate(y=segment)
        mean_zcr_segment.append(np.mean(zcr))
        lbl_mean_zcr_segment.append(f'mean_zcr_s{i}')

        # Calcular el "mean amplitude" de cada segmento
        mean_amp_segment.append(np.mean(np.abs(segment)))
        lbl_mean_amp_segment.append(f'mean_amp_s{i}')

        # Calcular el "max amplitude" de cada segmento
        max_amp_segment.append(np.max(np.abs(segment)))
        lbl_max_amp_segment.append(f'max_amp_s{i}')

        # Calcular los coeficientes MFCC
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)

        # Calcular el "mean mfcc_1" de cada segmento
        mean_mfcc_1_segment.append(np.mean(mfcc[0]))
        lbl_mean_mfcc_1_segment.append(f'mean_mfcc_1_s{i}')

        # Calcular el "mean mfcc_2" de cada segmento
        mean_mfcc_2_segment.append(np.mean(mfcc[1]))
        lbl_mean_mfcc_2_segment.append(f'mean_mfcc_2_s{i}')

        # Calcular el "mean mfcc_5" de cada segmento
        mean_mfcc_5_segment.append(np.mean(mfcc[4]))
        lbl_mean_mfcc_5_segment.append(f'mean_mfcc_5_s{i}')

        # Calcular el "max mfcc_2" de cada segmento
        max_mfcc_2_segment.append(np.max(mfcc[1]))
        lbl_max_mfcc_2_segment.append(f'max_mfcc_2_s{i}')

        # Calcular el "max mfcc_4" de cada segmento
        max_mfcc_4_segment.append(np.max(mfcc[3]))
        lbl_max_mfcc_4_segment.append(f'max_mfcc_4_s{i}')

        # Calcular el "max mfcc_5" de cada segmento
        max_mfcc_5_segment.append(np.max(mfcc[4]))
        lbl_max_mfcc_5_segment.append(f'max_mfcc_5_s{i}')

        # Calcular el "std mfcc_2" de cada segmento
        std_mfcc_2_segment.append(np.std(mfcc[1]))
        lbl_std_mfcc_2_segment.append(f'std_mfcc_2_s{i}')

        # Calcular el "std mfcc_4" de cada segmento
        std_mfcc_4_segment.append(np.std(mfcc[3]))
        lbl_std_mfcc_4_segment.append(f'std_mfcc_4_s{i}')

        # Calcular el "std mfcc_5" de cada segmento
        std_mfcc_5_segment.append(np.std(mfcc[4]))
        lbl_std_mfcc_5_segment.append(f'std_mfcc_5_s{i}')

        # Calcular el "mean spectral centroid" de cada segmento
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        mean_spectral_centroid_segment.append(np.mean(spectral_centroid))
        lbl_mean_spectral_centroid_segment.append(f'mean_spectral_centroid_s{i}')

        # Calcular el "mean spectral bandwidth" de cada segmento
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        mean_spectral_bandwidth_segment.append(np.mean(spectral_bandwidth))
        lbl_mean_spectral_bandwidth_segment.append(f'mean_spectral_bandwidth_s{i}')


    features.extend([n_samples, max_zcr_total])
    features.extend(mean_zcr_segment)
    features.extend(mean_amp_segment)
    features.extend(max_amp_segment)
    features.extend(mean_mfcc_1_segment)
    features.extend(mean_mfcc_2_segment)
    features.extend(mean_mfcc_5_segment)
    features.extend(max_mfcc_2_segment)
    features.extend(max_mfcc_4_segment)
    features.extend(max_mfcc_5_segment)
    features.extend(std_mfcc_2_segment)
    features.extend(std_mfcc_4_segment)
    features.extend(std_mfcc_5_segment)
    features.extend(mean_spectral_centroid_segment)
    features.extend(mean_spectral_bandwidth_segment)

    feature_labels.extend(lbl_mean_zcr_segment)
    feature_labels.extend(lbl_mean_amp_segment)
    feature_labels.extend(lbl_max_amp_segment)
    feature_labels.extend(lbl_mean_mfcc_1_segment)
    feature_labels.extend(lbl_mean_mfcc_2_segment)
    feature_labels.extend(lbl_mean_mfcc_5_segment)
    feature_labels.extend(lbl_max_mfcc_2_segment)
    feature_labels.extend(lbl_max_mfcc_4_segment)
    feature_labels.extend(lbl_max_mfcc_5_segment)
    feature_labels.extend(lbl_std_mfcc_2_segment)
    feature_labels.extend(lbl_std_mfcc_4_segment)
    feature_labels.extend(lbl_std_mfcc_5_segment)
    feature_labels.extend(lbl_mean_spectral_centroid_segment)
    feature_labels.extend(lbl_mean_spectral_bandwidth_segment)
    
    return features, feature_labels

# -----------------------------------------------------------------------------------

# Extracción de caracteristicas -----------------------------------------------------

# Path base de datos procesada
processed_path = '/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/audios/nuevos_audios'

# Etiquetas
labels = ['banana','manzana','naranja','pera']

X = []
y = []
all_feature_labels = []

# Se recorren las carpetas de las etiquetas
for label_index, label in enumerate(labels):
    folder = os.path.join(processed_path, label)

    for audio_file in os.listdir(folder):
        file_path = os.path.join(folder, audio_file)
        if file_path.endswith('.wav'):
            # Extraer características
            features, feature_labels = extract_features(file_path)
            X.append(features)
            y.append(label_index)
            if not all_feature_labels:
                all_feature_labels = feature_labels

# Convertir a arrays de Numpy
X = np.array(X)
y = np.array(y)

# Mostrar las formas de los arrays
print(X.shape)
print(y.shape)

# -----------------------------------------------------------------------------------

# Selección de caracteristicas ------------------------------------------------------
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)

# MinMax Scaler
X_scaled = (X - min_values) / (max_values - min_values + 1e-10)

filas = X_scaled.shape[0]
columnas = X_scaled.shape[1]
data = np.zeros((5, columnas))

# Calcular la varianza general y por clase
for i in range(columnas):
    filas = X_scaled.shape[0] 
    clases = 4
    div_filas = filas // clases
    banana = X_scaled[0:div_filas, i]
    manzana = X_scaled[div_filas:div_filas*2, i]
    naranja = X_scaled[div_filas*2:div_filas*3, i]
    pera = X_scaled[div_filas*3:div_filas*4, i]
    var_columna = np.var(X_scaled[:, i])
    var_banana = np.var(banana)
    var_manzana = np.var(manzana)
    var_naranja = np.var(naranja)
    var_pera = np.var(pera)
    data[0, i] = var_columna
    data[1, i] = var_banana
    data[2, i] = var_manzana
    data[3, i] = var_naranja
    data[4, i] = var_pera

# Calcular la relación entre la varianza general y la varianza por clase
relacion_varianza = np.zeros(columnas)
for i in range(columnas):
    var_general = data[0, i]
    var_clases = np.mean(data[1:, i])
    relacion_varianza[i] = var_general / (var_clases + 1e-10)  

# Asignar puntuaciones basadas en la relación de varianza
puntuaciones = np.zeros((2, columnas))
for i in range(columnas):
    puntuaciones[0, i] = i
    puntuaciones[1, i] = relacion_varianza[i]

# Ordenar las características de mayor a menor puntuación
puntuaciones = puntuaciones[:, puntuaciones[1, :].argsort()[::-1]]

# Seleccionar las n características con mayor puntuación
n = 13
X_selected = X_scaled[:, puntuaciones[0, 0:n].astype(int)]

# Obtener los indices de las características con mayor puntuación 
selected_indices = puntuaciones[0, 0:n].astype(int)  
selected_features = [all_feature_labels[i] for i in selected_indices]
print(f'Índices de características seleccionadas: {selected_indices}')
print(f'Características seleccionadas: {selected_features}')

# Obtener los valores maximos y minimos de las características seleccionadas
max_values = max_values[selected_indices]
min_values = min_values[selected_indices]

print(max_values.shape)
print(min_values.shape)

# -----------------------------------------------------------------------------------

# # Selección de características utilizando SelectKBest
# n = 13  # Número de características a seleccionar
# selector = SelectKBest(score_func=f_classif, k=n)
# X_selected = selector.fit_transform(X, y)

# # Obtener los índices de las características seleccionadas
# selected_indices = selector.get_support(indices=True)
# selected_features = [all_feature_labels[i] for i in selected_indices]

# print(f'Índices de características seleccionadas: {selected_indices}')
# print(f'Características seleccionadas: {selected_features}')

# -----------------------------------------------------------------------------------

# Visualización - PCA ---------------------------------------------------------------

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_selected)

# Plot 2D de las características seleccionadas
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='yellow', label='Banana')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Manzana')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], c='orange', label='Naranja')
plt.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], c='green', label='Pera')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Características de Audio - PCA 2D')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Plot 3D de las características seleccionadas
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], X_pca[y == 0, 2], c='yellow', label='Banana')
ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], X_pca[y == 1, 2], c='red', label='Manzana')
ax.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], X_pca[y == 2, 2], c='orange', label='Naranja')
ax.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], X_pca[y == 3, 2], c='green', label='Pera')
ax.set_xlabel('Componente principal 1')
ax.set_ylabel('Componente principal 2')
ax.set_zlabel('Componente principal 3')
plt.title('Características de Audio - PCA 3D')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------

# Exportar las características seleccionadas ----------------------------------------

# save_path = '/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/audios/audio_features/test'

# # Guardar las características seleccionadas
# np.save(os.path.join(save_path, 'X.npy'), X_selected)
# np.save(os.path.join(save_path, 'y.npy'), y)
# # np.save(os.path.join(save_path, "selected_indices.npy"), selected_indices)
# np.save(os.path.join(save_path, 'max_values.npy'), max_values)
# np.save(os.path.join(save_path, 'min_values.npy'), min_values)

# -----------------------------------------------------------------------------------

