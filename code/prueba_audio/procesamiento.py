import os
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt

# ------------------------------------ Funciones ------------------------------------

# Función para crear un filtro pasa-bajo
def low_pass_filter(data, cutoff_freq, sample_rate):
    nyquist_rate = sample_rate / 2.0
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

# Función para crear un filtro pasa-alto
def high_pass_filter(data, cutoff_freq, sample_rate):
    nyquist_rate = sample_rate / 2.0
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

def process_audio(input_path, output_path):
    # Cargar el audio
    y, sr = librosa.load(input_path, sr=44100)

    # Aplicar filtro pasa-banda
    low_cutoff = 300 
    high_cutoff = 5000
    y_filtered = low_pass_filter(y, high_cutoff, sr)
    y_filtered = high_pass_filter(y_filtered, low_cutoff, sr)

    # Cortar el audio
    trimmed_audio, _ = librosa.effects.trim(y_filtered, top_db=25)

    # Normalizar el audio
    normalized_audio = trimmed_audio / np.max(np.abs(trimmed_audio))

    # Guardar el audio recortado y normalizado
    sf.write(output_path, normalized_audio, sr)

# -----------------------------------------------------------------------------------

# Path a la carpeta en donde se encuentran los audios de la base de datos sin procesar
base_path = '/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/audios/test_originales'

# Path base de datos procesada
processed_path = '/Users/martinarobyculasso/Desktop/V3_Trabajo_Final_IA1/dataset/audios/test_procesados'

# Etiquetas
labels = ['banana','manzana','naranja','pera']

# Se recorren las carpetas de las etiquetas
for label in labels:
    # Se crea la carpeta de la etiqueta en la base de datos procesada (si no existe)
    os.makedirs(os.path.join(processed_path, label), exist_ok=True)

    # Se recorren los archivos de audio de la etiqueta
    for audio_file in os.listdir(os.path.join(base_path, label)):
        if audio_file.endswith('.wav'):
            # Se procesa el audio
            process_audio(
                os.path.join(base_path, label, audio_file),
                os.path.join(processed_path, label, audio_file)
            )