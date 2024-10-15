import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import soundfile as sf
import scipy.signal as signal
from sklearn.decomposition import PCA


class ClasificaAudio:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(base_path, '../../dataset/audios/audio_features')
        self.X_dataset = np.load(os.path.join(dataset_path, 'X.npy'))
        self.y_dataset = np.load(os.path.join(dataset_path, 'y.npy'))
        self.max_dataset_values = np.load(os.path.join(dataset_path, 'max_values.npy'))
        self.min_dataset_values = np.load(os.path.join(dataset_path, 'min_values.npy'))
        self.labels = ['banana', 'manzana', 'naranja', 'pera']
        self.pca = PCA(n_components=3)
        self.X_pca = self.pca.fit_transform(self.X_dataset)
        self.audio_obj = None

    def set_audio(self, audio_obj):
        self.audio_obj = audio_obj

    # Función para crear un filtro pasa-bajo
    def low_pass_filter(self, data, cutoff_freq, sample_rate):
        nyquist_rate = sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist_rate
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data

    # Función para crear un filtro pasa-alto
    def high_pass_filter(self, data, cutoff_freq, sample_rate):
        nyquist_rate = sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist_rate
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        filtered_data = signal.lfilter(b, a, data)
        return filtered_data

    def process_audio(self):
        initial_cut_duration = 0.25
        initial_cut_samples = int(initial_cut_duration * self.audio_obj.sr)

        if len(self.audio_obj.audio) > initial_cut_samples:
            self.audio_obj.audio = self.audio_obj.audio[initial_cut_samples:]       # Cortar los primeros 0.25 segundos de audio (ruido de tecla)

        trimmed_audio, _ = librosa.effects.trim(self.audio_obj.audio, top_db=25)    # Cortar partes silenciosas del audio

        # Verificar el número de muestras para saber si se ha podido cortar el audio correctamente
        max_samples = 39600
        if len(trimmed_audio) > max_samples:
            raise ValueError(f"No se ha podido procesar el audio. Por favor, vuelva a intentarlo.")

        low_cutoff = 300
        high_cutoff = 5000
        y_low_filtered = self.low_pass_filter(trimmed_audio, low_cutoff, self.audio_obj.sr)         # Aplicar filtro pasa-bajo
        y_high_filtered = self.high_pass_filter(y_low_filtered, high_cutoff, self.audio_obj.sr)     # Aplicar filtro pasa-alto

        normalized_audio = y_high_filtered / np.max(np.abs(y_high_filtered))                        # Normalizar amplitud del audio

        self.audio_obj.audio = normalized_audio                                                     # Actualizar audio

        sf.write(self.audio_obj.path, normalized_audio, self.audio_obj.sr)

    def extract_features(self):
        y = self.audio_obj.audio
        sr = self.audio_obj.sr

        features = []

        n_samples = len(y)                              # Número de muestras
        max_zcr_total = np.max(
            librosa.feature.zero_crossing_rate(y=y))    # Máximo valor de la tasa de cruce por cero del audio compleyo

        # Dividir el audio en 10 partes iguales
        segment_length = len(y) // 10
        segments = [y[i * segment_length:(i + 1) * segment_length] for i in range(10)]

        i = 0
        for segment in segments:
            i += 1

            if i == 2:
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
                mean_mfcc_5_s2 = np.std(mfcc[4])                                # Desviación estándar de los coeficientes MFCC 5 del segmento 2

            elif i == 3:
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
                max_mfcc_2_s3 = np.max(mfcc[1])                                 # Máximo valor de los coeficientes MFCC 2 del segmento 3

                mean_amp_s3 = np.mean(np.abs(segment))                          # Valor medio de la amplitud del segmento 3

            elif i == 4:
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
                mean_mfcc_1_s4 = np.mean(mfcc[0])                               # Valor medio de los coeficientes MFCC 1 del segmento 4

                mean_amp_s4 = np.mean(np.abs(segment))                          # Valor medio de la amplitud del segmento 4

                max_amp_s4 = np.max(np.abs(segment))                            # Máximo valor de la amplitud del segmento 4

            elif i == 5:
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
                max_mfcc_4_s5 = np.max(mfcc[3])                                 # Máximo valor de los coeficientes MFCC 4 del segmento 5
                mean_mfcc_5_s5 = np.mean(mfcc[4])                               # Valor medio de los coeficientes MFCC 5 del segmento 5

            elif i == 7:
                mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=5)
                mean_mfcc_1_s7 = np.mean(mfcc[0])                               # Valor medio de los coeficientes MFCC 1 del segmento 7

                max_amp_s7 = np.max(np.abs(segment))                            # Máximo valor de la amplitud del segmento 7

            elif i == 8:
                zcr = librosa.feature.zero_crossing_rate(y=segment)
                mean_zcr_s8 = np.mean(zcr)                                      # Valor medio de la tasa de cruce por cero del segmento 8

        features = [max_zcr_total, mean_mfcc_1_s4, mean_amp_s4, max_amp_s4, max_mfcc_4_s5, n_samples, max_mfcc_2_s3, mean_mfcc_1_s7, mean_mfcc_5_s5, mean_mfcc_5_s2, mean_zcr_s8, max_amp_s7, mean_amp_s3]

        # Escalar las características con Min-Max Scaling
        scaled_features = (features - self.min_dataset_values) / (self.max_dataset_values - self.min_dataset_values + 1e-10)

        self.audio_obj.set_features(scaled_features)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self):
        X_train = self.X_dataset
        y_train = self.y_dataset
        X_new = self.audio_obj.get_features()
        k = 7   # Número de vecinos más cercanos

        # Algortimo de clasificación k-NN 

        # Se calculan las distancias entre el punto a clasificar y los puntos de entrenamiento
        distancias = []
        for i in range(len(X_train)):
            distancias.append(self.euclidean_distance(X_train[i], X_new))
        distancias = np.array(distancias)

        # Se ordenan las distancias de menor a mayor
        distancias_sorted_indices = np.argsort(distancias)

        # Se obtienen los índices de los k vecinos más cercanos
        k_indices = distancias_sorted_indices[:k]

        # Se obtienen las etiquetas de los k vecinos más cercanos
        k_etiquetas = y_train[k_indices]

        # Se obtiene la etiqueta más común
        etiqueta = np.bincount(k_etiquetas).argmax()

        self.audio_obj.set_predicted_label(self.labels[etiqueta])
        return etiqueta

    def plot_X_new(self):
        X_new_pca = self.pca.transform([self.audio_obj.get_features()])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        colors = ['yellow', 'red', 'orange', 'green']
        labels = ['Banana', 'Manzana', 'Naranja', 'Pera']

        for i, color in enumerate(colors):
            ax.scatter(self.X_pca[self.y_dataset == i, 0],
                       self.X_pca[self.y_dataset == i, 1],
                       self.X_pca[self.y_dataset == i, 2],
                       c=color, label=labels[i])

        ax.scatter(X_new_pca[0, 0], X_new_pca[0, 1], X_new_pca[0, 2], c='black', marker='x', s=100,
                   label='Audio Clasificado')

        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        ax.set_title('Características de Audio - PCA')
        ax.grid(True)
        ax.legend()

        plt.show()
