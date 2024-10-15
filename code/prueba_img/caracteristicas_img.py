import cv2
import os
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# ------------------------------------ Funciones ------------------------------------

def eliminar_fondo_azul(image):
    # Aplicar desenfoque gaussiano
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color azul en HSV
    lower_blue = np.array([90, 10, 0])      # Incluye tonos muy oscuros y poco saturados (para considerar sombras)
    upper_blue = np.array([130, 255, 255])  # Máximo del rango azul

    # Crear la máscara para eliminar el color azul
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Invertir la máscara para obtener el área que NO es azul
    mask_inv = cv2.bitwise_not(mask)

    # Aplicar la máscara para eliminar el fondo azul
    image_sin_fondo = cv2.bitwise_and(image, image, mask=mask_inv)

    return image_sin_fondo, mask_inv

def eliminar_fondo_blanco(image):
    # Aplicar desenfoque gaussiano
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango de color blanco en HSV
    lower_white = np.array([0, 0, 200])      # Límite inferior del blanco
    upper_white = np.array([180, 30, 255])   # Límite superior del blanco

    # Crear la máscara para eliminar el color blanco
    mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Invertir la máscara para obtener el área que NO es blanco
    mask_inv = cv2.bitwise_not(mask)

    # Aplicar la máscara para eliminar el fondo blanco
    image_sin_fondo = cv2.bitwise_and(image, image, mask=mask_inv)

    return image_sin_fondo, mask_inv

def detectar_color_fondo(image):
    # Convertir la imagen a espacio de color HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir los rangos de color para azul y blanco
    lower_blue = np.array([90, 10, 0])
    upper_blue = np.array([130, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    
    # Crear máscaras para azul y blanco
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # Contar los píxeles en cada máscara
    count_blue = np.sum(mask_blue)
    count_white = np.sum(mask_white)
    
    # Determinar el color predominante del fondo
    if count_blue > count_white:
        return 'blue'
    else:
        return 'white'

def procesar_imagen_y_extraer_caracteristicas(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return None
    
    # Obtener dimensiones originales de la imagen
    height, width = image.shape[:2]
    
    # Redimensionar la imagen dividiendo las dimensiones originales por el factor de escala (4)
    if height < 500 or width < 500:
        new_size = (width, height)
    else:
        new_size = (width // 5, height // 5)
        
    image = cv2.resize(image, new_size)

    # Detectar el color del fondo
    color_fondo = detectar_color_fondo(image)
    
    # Eliminar el fondo adecuado
    if color_fondo == 'blue':
        image_sin_fondo, mask = eliminar_fondo_azul(image)
    else:
        image_sin_fondo, mask = eliminar_fondo_blanco(image)

    # Convertir la imagen enmascarada a HSV
    hsv_image = cv2.cvtColor(image_sin_fondo, cv2.COLOR_BGR2HSV)
    
    # Extraer los píxeles correspondientes a la fruta
    pixels = hsv_image.reshape(-1, 3)
    pixels = pixels[mask.flatten() != 0]            # Eliminar los píxeles del fondo

    if len(pixels) == 0:
        print("No se encontraron píxeles de la fruta.")
        return None

    # Extraer los valores de tono (hue) de los píxeles
    hues = pixels[:, 0]
    
    # Calcular el tono más frecuente (color predominante)
    hue_counts = Counter(hues)
    hue_predominante = hue_counts.most_common(1)[0][0]

    # Encontrar los contornos de la fruta
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No se encontró contorno.")
        return None

    # Suponer que el contorno más grande es la fruta
    contorno_mayor = max(contours, key=cv2.contourArea)

    # Calcular el área y el perímetro del contorno
    area = cv2.contourArea(contorno_mayor)
    perimetro = cv2.arcLength(contorno_mayor, True)

    if perimetro == 0:
        print("El perímetro es cero, no se puede calcular la circularidad.")
        return None

    # Calcular la circularidad
    circularidad = (4 * np.pi * area) / (perimetro ** 2)

    # Calcular la relación de aspecto (aspect ratio)
    x, y, w, h = cv2.boundingRect(contorno_mayor)   # Rectángulo delimitador
    relacion_aspecto = float(w) / h                 # Ancho / Alto

    # Calcular el centroide del contorno
    M = cv2.moments(contorno_mayor)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Definir el tamaño del cuadrado
    lado = 16
    half_lado = lado // 2

    # Definir las coordenadas del cuadrado alrededor del centroide
    x_start = max(cX - half_lado, 0)
    y_start = max(cY - half_lado, 0)
    x_end = min(cX + half_lado, image_sin_fondo.shape[1])
    y_end = min(cY + half_lado, image_sin_fondo.shape[0])

    # Asegurarse de que el cuadrado esté dentro del contorno
    if x_end - x_start < lado:
        if x_start == 0:
            x_end = x_start + lado
        else:
            x_start = x_end - lado

    if y_end - y_start < lado:
        if y_start == 0:
            y_end = y_start + lado
        else:
            y_start = y_end - lado

    # Extraer los píxeles dentro del cuadrado
    cuadrado = image_sin_fondo[y_start:y_end, x_start:x_end]

    # Calcular el color RGB promedio
    rgb_promedio = np.mean(cuadrado, axis=(0, 1))

    # Calcular los momentos de Hu
    momentos = cv2.moments(contorno_mayor)
    momentos_hu = cv2.HuMoments(momentos).flatten()

    # Tomar el logaritmo de los momentos de Hu
    momentos_hu = -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu) + 1e-10)

    # # Mostrar el contorno de la fruta
    # imagen_contorno = image_sin_fondo.copy()
    # cv2.drawContours(imagen_contorno, [contorno_mayor], -1, (0, 255, 0), 2)
    # cv2.imshow('Contorno de la Fruta', imagen_contorno)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Devolver las características calculadas
    return [rgb_promedio[2], rgb_promedio[1], rgb_promedio[0], circularidad, relacion_aspecto, hue_predominante] + momentos_hu.tolist()

# -----------------------------------------------------------------------------------

# Extracción de caracteristicas -----------------------------------------------------

# Ruta base de las imágenes
base_path = '/Users/martinarobyculasso/Desktop/blue_img'

# Etiquetas
labels = ['banana', 'manzana', 'naranja', 'pera']

X = []
y = []
all_feature_labels = []

# Recorrer las carpetas y procesar imágenes
for label_index, label in enumerate(labels):
    folder_path = os.path.join(base_path, label)
    
    # Verificar si el directorio existe
    if not os.path.isdir(folder_path):
        print(f"Directorio no encontrado: {folder_path}")
        continue
    
    # Listar archivos en el directorio de la fruta correspondiente
    for filename in os.listdir(folder_path):
        # Ignorar archivos que comiencen con '.'
        if filename.startswith('.'):
            continue

        # Construir la ruta completa de la imagen
        image_path = os.path.join(folder_path, filename)
        print(f"\nProcesando imagen: {image_path}")
        
        # Procesar la imagen y extraer características
        caracteristicas = procesar_imagen_y_extraer_caracteristicas(image_path)
        
        if caracteristicas is not None:
            # Almacenar características junto con la etiqueta
            X.append(caracteristicas)
            y.append(label_index)
            if not all_feature_labels:
                all_feature_labels = ['rgb1', 'rgb2', 'rgb3', 'circularidad', 'relacion_aspecto', 'hue_predominante', 'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7']

# Convertir a arrays de Numpy
X = np.array(X)
y = np.array(y)

# Mostrar las formas de los arrays
print(X.shape)
print(y.shape)

# -----------------------------------------------------------------------------------

# Selección de caracteristicas ------------------------------------------------------

# Escalar las características (min-max)
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
X_scaled = (X - min_values) / (max_values - min_values + 1e-10)

columnas = X_scaled.shape[1]
data = np.zeros((5, columnas))

# Agrupar los índices por clase
indices_banana = np.where(y == 0)[0]
indices_manzana = np.where(y == 1)[0]
indices_naranja = np.where(y == 2)[0]
indices_pera = np.where(y == 3)[0]

# Calcular la varianza general y por clase
for i in range(columnas):
    var_columna = np.var(X_scaled[:, i])
    var_banana = np.var(X_scaled[indices_banana, i])
    var_manzana = np.var(X_scaled[indices_manzana, i])
    var_naranja = np.var(X_scaled[indices_naranja, i])
    var_pera = np.var(X_scaled[indices_pera, i])
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
n = 5   
X_selected = X_scaled[:, puntuaciones[0, 0:n].astype(int)]

# Obtener los indices de las características con mayor puntuación 
selected_indices = puntuaciones[0, 0:n].astype(int)  
selected_features = [all_feature_labels[i] for i in selected_indices]
print(f'Índices de características seleccionadas: {selected_indices}')
print(f'Características seleccionadas: {selected_features}')

# Obtener los valores maximos y minimos de las características seleccionadas
max_values = max_values[selected_indices]
min_values = min_values[selected_indices]

# -----------------------------------------------------------------------------------

# Visualización - PCA ---------------------------------------------------------------

# Crear un objeto PCA con 2 componentes
pca = PCA(n_components=2)

# Aplicar PCA a las características seleccionadas
X_pca = pca.fit_transform(X_selected)

# Plot 2D de las características seleccionadas
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='yellow', label='Banana')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='red', label='Manzana')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], c='orange', label='Naranja')
plt.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], c='green', label='Pera')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Características de Imagen - PCA 2D')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------------------------

# Exportar las características seleccionadas ----------------------------------------

# save_path = '/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/img/img_features'

# # Guardar las características seleccionadas
# np.save(os.path.join(save_path, 'X.npy'), X_selected)
# np.save(os.path.join(save_path, 'y.npy'), y)
# # np.save(os.path.join(save_path, "selected_indices.npy"), selected_indices)
# np.save(os.path.join(save_path, 'max_values.npy'), max_values)
# np.save(os.path.join(save_path, 'min_values.npy'), min_values)

# -----------------------------------------------------------------------------------
