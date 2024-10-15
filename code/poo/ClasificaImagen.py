import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Tengo un problema con el backend por defecto (Tk), así que lo cambio a Qt5Agg
from collections import Counter
from sklearn.decomposition import PCA


class ClasificaImagen:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, '../../dataset/img/img_features')
        self.X_dataset = np.load(os.path.join(dataset_path, 'X.npy'))
        self.y_dataset = np.load(os.path.join(dataset_path, 'y.npy'))
        self.max_dataset_values = np.load(os.path.join(dataset_path, 'max_values.npy'))
        self.min_dataset_values = np.load(os.path.join(dataset_path, 'min_values.npy'))
        self.centroides = np.load(os.path.join(dataset_path, 'centroides.npy'))
        self.etiquetas_centroides = np.load(os.path.join(dataset_path, 'etiquetas_centroides.npy'))
        self.labels = ['banana', 'manzana', 'naranja', 'pera']
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_dataset)
        self.centroides_pca = self.pca.transform(self.centroides)
    
    def eliminar_fondo(self, image):

        # Aplicar desenfoque gaussiano
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Convertir la imagen a espacio de color HSV
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Definir el rango de color azul en HSV
        lower_blue = np.array([90, 10, 0])          # Incluye tonos muy oscuros y poco saturados (para considerar sombras)
        upper_blue = np.array([130, 255, 255])      # Máximo del rango azul

        # Crear la máscara para eliminar el color azul
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Invertir la máscara para obtener el área que NO es azul
        mask_inv = cv2.bitwise_not(mask)

        # Aplicar la máscara para eliminar el fondo azul
        image_sin_fondo = cv2.bitwise_and(image, image, mask=mask_inv)

        return image_sin_fondo, mask_inv

    def extract_features(self, img_obj):
        
        image = cv2.imread(img_obj.path)
        height, width = image.shape[:2]
        
        # Redimensionar la imagen dividiendo las dimensiones originales por el factor de escala (5)
        if height < 500 or width < 500:
            new_size = (width, height)
        else:
            new_size = (width // 5, height // 5)
            
        image = cv2.resize(image, new_size)
        
        # Eliminar el fondo 
        image_sin_fondo, mask = self.eliminar_fondo(image)

        # Convertir la imagen enmascarada a HSV
        hsv_image = cv2.cvtColor(image_sin_fondo, cv2.COLOR_BGR2HSV)
        
        # Extraer los píxeles correspondientes a la fruta
        pixels = hsv_image.reshape(-1, 3)
        pixels = pixels[mask.flatten() != 0]

        if len(pixels) == 0:
            print("No se encontraron píxeles de la fruta.")
            return None

        # Extraer los valores de tono (hue) de los píxeles
        hues = pixels[:, 0]
        
        # Calcular el tono más frecuente (color predominante)
        hue_counts = Counter(hues)
        hue_predominante = hue_counts.most_common(1)[0][0]         # Hue más frecuente

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
        circularidad = (4 * np.pi * area) / (perimetro ** 2)        # Circularidad 

        # Calcular los momentos de Hu
        momentos = cv2.moments(contorno_mayor)
        momentos_hu = cv2.HuMoments(momentos).flatten()
        momentos_hu = -np.sign(momentos_hu) * np.log10(np.abs(momentos_hu) + 1e-10)

        hu1 = momentos_hu[0]                                        # Momento de Hu 1
        hu3 = momentos_hu[2]                                        # Momento de Hu 3
        hu4 = momentos_hu[3]                                        # Momento de Hu 4

        features = np.array([hu1, circularidad, hu3, hu4, hue_predominante])

        scaled_features = (features - self.min_dataset_values) / (self.max_dataset_values - self.min_dataset_values + 1e-10)

        img_obj.set_features(scaled_features)
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, img_obj):
        X_new = img_obj.get_features()
        k = 4       # Número de clusters

        # Algorigmo de clasificación KMeans

        # Calcular la distancia entre el nuevo punto y los centroides almacenados
        distancias = np.array([self.euclidean_distance(X_new, centroide) for centroide in self.centroides]).T

        # Asignar el nuevo punto al cluster más cercano
        cluster = np.argmin(distancias)

        img_obj.set_predicted_label(self.labels[self.etiquetas_centroides[cluster]])
 

    def plot_new_samples(self, image_objects):

        plt.figure(figsize=(10, 7))

        colors = ['yellow', 'red', 'orange', 'green']
        labels = ['Banana', 'Manzana', 'Naranja', 'Pera']

        # Graficar los puntos del dataset
        for i, color in enumerate(colors):
            plt.scatter(self.X_pca[self.y_dataset == i, 0], self.X_pca[self.y_dataset == i, 1], c=color, label=labels[i])

        # Graficar los centroides
        for i, color in enumerate(colors):
            color = colors[self.etiquetas_centroides[i]]
            plt.scatter(self.centroides_pca[i, 0], self.centroides_pca[i, 1], c=color, marker='x', s=200)
            plt.text(self.centroides_pca[i, 0], self.centroides_pca[i, 1], f'Centroide {labels[self.etiquetas_centroides[i]]}', fontsize=9, ha='right', va='top')

        # Graficar los nuevos puntos
        for idx, img_obj in enumerate(image_objects):
            e_feats = img_obj.get_features()
            X_new_pca = self.pca.transform([e_feats])  # Asegúrate de que las características se pasen en forma de matriz
            estante_label = f'Estante {idx + 1}'
            plt.scatter(X_new_pca[0, 0], X_new_pca[0, 1], c='black', marker='x', s=100)
            plt.text(X_new_pca[0, 0], X_new_pca[0, 1], estante_label, fontsize=9, ha='right')

        # Agregar el ítem "Estantes" con una cruz negra a la leyenda
        plt.scatter([], [], c='black', marker='x', label='Estantes')

        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Características de Imágenes - PCA')
        plt.grid(True)
        plt.legend()
        plt.show()




        