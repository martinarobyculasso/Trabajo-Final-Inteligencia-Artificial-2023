import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns

def eliminar_fondo(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 10, 0])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask)
    image_sin_fondo = cv2.bitwise_and(image, image, mask=mask_inv)
    return image_sin_fondo, mask_inv

def extract_features(path):
    image = cv2.imread(path)
    height, width, _ = image.shape
    if height < 500 or width < 500:
        new_size = (width, height)
    else:
        new_size = (width // 5, height // 5)
    image = cv2.resize(image, new_size)
    image_sin_fondo, mask = eliminar_fondo(image)
    hsv_image = cv2.cvtColor(image_sin_fondo, cv2.COLOR_BGR2HSV)
    pixels = hsv_image.reshape(-1, 3)
    pixels = pixels[mask.flatten() != 0]
    if len(pixels) == 0:
        return None
    hues = pixels[:, 0]
    hue_counts = Counter(hues)
    hue_predominante = hue_counts.most_common(1)[0][0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    contorno_mayor = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contorno_mayor)
    perimetro = cv2.arcLength(contorno_mayor, True)
    if perimetro == 0:
        return None
    circularidad = (4 * np.pi * area) / (perimetro ** 2)
    momentos = cv2.moments(contorno_mayor)
    momentos_hu = cv2.HuMoments(momentos).flatten()
    momentos_hu = -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu) + 1e-10)
    hu1 = momentos_hu[0]
    hu3 = momentos_hu[2]
    hu4 = momentos_hu[3]
    features = [hu1, circularidad, hu3, hu4, hue_predominante]

    imagen_contorno = image_sin_fondo.copy()
    cv2.drawContours(imagen_contorno, [contorno_mayor], -1, (0, 255, 0), 2)
    cv2.imshow('Contorno de la Fruta', imagen_contorno)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return features

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict(features, centroides):
    distancias = np.array([euclidean_distance(features, centroide) for centroide in centroides]).T
    cluster = np.argmin(distancias)
    return etiquetas_centroides[cluster]

# Cargar puntos y etiquetas
X = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/X.npy')
y = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/y.npy', allow_pickle=True)

# Cargar centroides y etiquetas
centroides = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/centroides.npy')
etiquetas_centroides = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/etiquetas_centroides.npy')

# Cargar minimos y maximos
minimos = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/min_values.npy')
maximos = np.load('/Users/martinarobyculasso/Desktop/V4_Trabajo_Final_IA1/dataset/img/img_features/max_values.npy')

# Directorio de imágenes de prueba
test_dir = '/Users/martinarobyculasso/Desktop/img_test_blue'

# Mapear nombres de carpetas a etiquetas
folder_to_label = {'banana': 0, 'manzana': 1, 'naranja': 2, 'pera': 3}

# Inicializar listas para etiquetas predichas y reales
y_true = []
y_pred = []

# Procesar cada imagen en el directorio de prueba
for folder_name in os.listdir(test_dir):
    if folder_name.startswith('.'):
        continue
    folder_path = os.path.join(test_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.startswith('.'):
                continue
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                features = (features - minimos) / (maximos - minimos)
                predicted_label = predict(features, centroides)
                y_true.append(folder_to_label[folder_name])
                y_pred.append(predicted_label)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['banana', 'manzana', 'naranja', 'pera'], yticklabels=['banana', 'manzana', 'naranja', 'pera'])
plt.xlabel('Etiqueta predicha')
plt.ylabel('Etiqueta verdadera')
plt.title('Matriz de confusión - Imágenes')
plt.show()