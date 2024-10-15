import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
import cv2
from PIL import Image, ImageDraw, ImageFont
import socket

from ClasificaAudio import ClasificaAudio
from Audio import Audio
from ClasificaImagen import ClasificaImagen
from Imagen import Imagen


class Controlador:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(base_path, '../../app')
        self.audio_path = os.path.join(app_path, 'audio_temp/audio_temp.wav')
        self.img_path = os.path.join(app_path, 'estantes')
        self.temp_img_path = os.path.join(app_path, 'img_temp/img_temp.jpg')
        self.estantes = ['estante_1', 'estante_2', 'estante_3', 'estante_4']
        self.estante_1 = None
        self.estante_2 = None
        self.estante_3 = None
        self.estante_4 = None
        self.marco = os.path.join(app_path, 'marco.png')
        self.font = os.path.join(app_path, 'FreeSans.otf')
        self.result = os.path.join(app_path, 'estantes_con_frutas.png')
        self.s = None
        self.is_recording = False
        self.audio_data = None
        self.samplerate = 44100
        self.clasificador_audio = ClasificaAudio()
        self.clasificador_imagen = ClasificaImagen()
        self.audio_log = []
        self.img_log = []

    # Audios -------------------------------------------------------------------

    def grabar_audio(self):
        self.is_recording = True
        self.audio_data = []

        with sd.InputStream(callback=self.audio_callback, samplerate=self.samplerate):
            sd.sleep(3000)
        self.is_recording = False

        audio_array = np.concatenate(self.audio_data, axis=0)
        sf.write(self.audio_path, audio_array, self.samplerate)

        audio_obj = Audio(self.audio_path)
        self.clasificador_audio.set_audio(audio_obj)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(str(status), file=sys.stderr)
        self.audio_data.append(indata.copy())

    def clasificar_audio(self):
        if self.clasificador_audio.audio_obj is None:
            raise ValueError("No se ha grabado ningún audio aún.")

        self.clasificador_audio.audio_obj.load_audio()
        self.clasificador_audio.process_audio()
        self.clasificador_audio.extract_features()
        prediction = self.clasificador_audio.predict()
        return self.clasificador_audio.audio_obj.get_predicted_label()

    def plot_audio(self):
        if self.clasificador_audio.audio_obj is None:
            raise ValueError("No se ha clasificado ningún audio aún.")

        self.clasificador_audio.plot_X_new()
    
    # --------------------------------------------------------------------------
   
    # Imágenes -----------------------------------------------------------------

    def tomar_fotos(self):
        # Inicia la cámara 
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("No se pudo acceder a la cámara.")

        saved_filenames = []

        for i, estante in enumerate(self.estantes):
            input(f"Presiona Enter para tomar la foto del {estante}...")

            # Captura una imagen
            ret, frame = cap.read()

            if not ret:
                print(f"No se pudo capturar la imagen del {estante}.")
                continue

            # Guarda la imagen en un archivo
            filename = os.path.join(self.img_path, f'{estante}.jpg')
            cv2.imwrite(filename, frame)
            saved_filenames.append(filename)

            # Muestra la imagen capturada
            cv2.imshow(f'Foto Capturada - {estante}', frame)
            cv2.waitKey(1000)  # Espera 1 segundo para mostrar la imagen

            # Cierra la ventana de visualización
            cv2.destroyAllWindows()

        # Libera la cámara
        cap.release()
        cv2.destroyAllWindows()

        self.cargar_fotos()

    def cargar_fotos(self):
        paths = []
        for estante in self.estantes:
            img_path = os.path.join(self.img_path, f'{estante}.jpg')
            paths.append(img_path)
        
        self.estante_1 = Imagen(paths[0])
        self.estante_2 = Imagen(paths[1])
        self.estante_3 = Imagen(paths[2])
        self.estante_4 = Imagen(paths[3])

    def clasificar_imagenes_estantes(self):
        # Extraer características y predecir para cada estante
        for estante in [self.estante_1, self.estante_2, self.estante_3, self.estante_4]:
            self.clasificador_imagen.extract_features(estante)
            self.clasificador_imagen.predict(estante)

        # Verificar que todos los estantes tengan etiquetas válidas
        estantes = [self.estante_1, self.estante_2, self.estante_3, self.estante_4]
        etiquetas = [estante.get_predicted_label() for estante in estantes]

        if any(etiqueta is None for etiqueta in etiquetas):
            print("Error: Al menos una etiqueta predicha es None.")
            return None

        return etiquetas
    
    def plot_images(self):
        estantes_obj = [self.estante_1, self.estante_2, self.estante_3, self.estante_4]
        self.clasificador_imagen.plot_new_samples(estantes_obj)

    # --------------------------------------------------------------------------

    # Resultados ---------------------------------------------------------------
    def obtener_estante(self):
        fruta_pedida = self.clasificador_audio.audio_obj.predicted_label

        if fruta_pedida == self.estante_1.get_predicted_label():
            return 1
        elif fruta_pedida == self.estante_2.get_predicted_label():
            return 2
        elif fruta_pedida == self.estante_3.get_predicted_label():
            return 3
        elif fruta_pedida == self.estante_4.get_predicted_label():
            return 4
        else:
            return None 

    def mostrar_estantes(self, fruta_pedida):
        if fruta_pedida is None:
            # Cargar imágenes
            base_img = Image.open(self.marco).convert("RGBA")
            fruta_e1 = Image.open(self.estante_1.path).convert("RGBA")
            fruta_e2 = Image.open(self.estante_2.path).convert("RGBA")
            fruta_e3 = Image.open(self.estante_3.path).convert("RGBA")
            fruta_e4 = Image.open(self.estante_4.path).convert("RGBA")

            # Redimensionar imágenes
            fruta_e1 = fruta_e1.resize((300, 300))
            fruta_e2 = fruta_e2.resize((300, 300))
            fruta_e3 = fruta_e3.resize((300, 300))
            fruta_e4 = fruta_e4.resize((300, 300))

            # Superponer imágenes
            base_img.paste(fruta_e1, (87, 106), fruta_e1)
            base_img.paste(fruta_e2, (458, 106), fruta_e2)
            base_img.paste(fruta_e3, (830, 106), fruta_e3)
            base_img.paste(fruta_e4, (1201, 106), fruta_e4)

            # Crear un objeto para dibujar en la imagen
            draw = ImageDraw.Draw(base_img) 

            # Añadir texto
            font = ImageFont.truetype(self.font, size=20)
            draw.text((100, 370), self.estante_1.get_predicted_label() , font=font, fill=(255, 255, 255, 255))
            draw.text((471, 370), self.estante_2.get_predicted_label(), font=font, fill=(255, 255, 255, 255))
            draw.text((844, 370), self.estante_3.get_predicted_label(), font=font, fill=(255, 255, 255, 255))
            draw.text((1215, 370), self.estante_4.get_predicted_label(), font=font, fill=(255, 255, 255, 255))

            # Mostrar la imagen
            base_img.show()
        else:
            fruta_pedida = fruta_pedida - 1

            # Cargar imágenes
            base_img = Image.open(self.marco).convert("RGBA")
            fruta_e1 = Image.open(self.estante_1.path).convert("RGBA")
            fruta_e2 = Image.open(self.estante_2.path).convert("RGBA")
            fruta_e3 = Image.open(self.estante_3.path).convert("RGBA")
            fruta_e4 = Image.open(self.estante_4.path).convert("RGBA")

            # Redimensionar imágenes
            fruta_e1 = fruta_e1.resize((300, 300))
            fruta_e2 = fruta_e2.resize((300, 300))
            fruta_e3 = fruta_e3.resize((300, 300))
            fruta_e4 = fruta_e4.resize((300, 300))

            # Superponer imágenes
            base_img.paste(fruta_e1, (87, 106), fruta_e1)
            base_img.paste(fruta_e2, (458, 106), fruta_e2)
            base_img.paste(fruta_e3, (830, 106), fruta_e3)
            base_img.paste(fruta_e4, (1201, 106), fruta_e4)

            # Crear un objeto para dibujar en la imagen
            draw = ImageDraw.Draw(base_img)

            # Dibujar un rectángulo alrededor de la fruta pedida
            if fruta_pedida == 0:
                draw.rectangle((87, 106, 387, 406), outline="green", width=8)
            elif fruta_pedida == 1:
                draw.rectangle((458, 106, 758, 406), outline="green", width=8)
            elif fruta_pedida == 2:
                draw.rectangle((830, 106, 1130, 406), outline="green", width=8)
            elif fruta_pedida == 3:
                draw.rectangle((1201, 106, 1501, 406), outline="green", width=8)
            
            # Añadir texto
            font = ImageFont.truetype(self.font, size=20)
            draw.text((100, 370), self.estante_1.get_predicted_label() , font=font, fill=(255, 255, 255, 255))
            draw.text((471, 370), self.estante_2.get_predicted_label(), font=font, fill=(255, 255, 255, 255))
            draw.text((844, 370), self.estante_3.get_predicted_label(), font=font, fill=(255, 255, 255, 255))
            draw.text((1215, 370), self.estante_4.get_predicted_label(), font=font, fill=(255, 255, 255, 255))

            # Guardar la imagen
            base_img.save(self.result)

            # Mostrar la imagen
            base_img.show()

    # --------------------------------------------------------------------------

    # Enviar resultados a través de WiFi ---------------------------------------

    def conectar(self):
        IP = '172.20.10.11'
        PORT = 8080
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((IP, PORT))
        except Exception as e:
            raise ValueError(f"Error al conectar con el ESP8266: {e}")

    def send(self, fruta, estante):
        if self.s is None:
            raise ValueError("Primero debes conectar antes de enviar datos.")
        try:
            # Convertir estante a string y concatenar con el carácter de fruta
            estante = str(estante)
            data = fruta[0] + estante               # Concatenar la fruta y el estante
            self.s.sendall(data.encode('utf-8'))    # Enviar los datos al ESP8266
        except Exception as e:
            raise ValueError(f"Error al enviar datos al ESP8266: {e}")

    def desconectar(self):
        if self.s is not None:
            self.s.close()
            self.s = None

    # --------------------------------------------------------------------------

    # Métodos para el modo de registro y reporte de audios ---------------------

    def registrar_audio(self):
        self.grabar_audio()
        predicted_label = self.clasificar_audio()
        return predicted_label

    def agregar_audio_log(self):
        self.audio_log.append(self.clasificador_audio.audio_obj)

    def reporte_audios(self):
        if len(self.audio_log) == 0:
            raise ValueError("Todavía no se han agregado audios al log.")

        total_audios = len(self.audio_log)
        correct_predictions = sum(1 for aud in self.audio_log if aud.real_label == aud.predicted_label)

        et_true = [aud.real_label for aud in self.audio_log]
        et_pred = [aud.predicted_label for aud in self.audio_log]

        conf_matrix = confusion_matrix(et_true, et_pred)

        etiquetas = sorted(set(et_true + et_pred))

        return total_audios, correct_predictions, conf_matrix, etiquetas

    def heat_map_audios(self, conf_matrix, etiquetas):
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
        plt.xlabel('Etiqueta predicha')
        plt.ylabel('Etiqueta real')
        plt.title('Matriz de Confusión')
        plt.show()

    def plot_session(self):
        if len(self.audio_log) == 0:
            raise ValueError("Todavía no se han agregado audios al log.")

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        X_pca = self.clasificador_audio.X_pca

        ax.scatter(X_pca[self.clasificador_audio.y_dataset == 0, 0],
                   X_pca[self.clasificador_audio.y_dataset == 0, 1],
                   X_pca[self.clasificador_audio.y_dataset == 0, 2],
                   c='yellow', label='Banana')
        ax.scatter(X_pca[self.clasificador_audio.y_dataset == 1, 0],
                   X_pca[self.clasificador_audio.y_dataset == 1, 1],
                   X_pca[self.clasificador_audio.y_dataset == 1, 2],
                   c='red', label='Manzana')
        ax.scatter(X_pca[self.clasificador_audio.y_dataset == 2, 0],
                   X_pca[self.clasificador_audio.y_dataset == 2, 1],
                   X_pca[self.clasificador_audio.y_dataset == 2, 2],
                   c='orange', label='Naranja')
        ax.scatter(X_pca[self.clasificador_audio.y_dataset == 3, 0],
                   X_pca[self.clasificador_audio.y_dataset == 3, 1],
                   X_pca[self.clasificador_audio.y_dataset == 3, 2],
                   c='green', label='Pera')

        for aud in self.audio_log:
            aud_pca = self.clasificador_audio.pca.transform([aud.features])
            color = 'purple' if aud.real_label == aud.predicted_label else 'red'
            ax.scatter(aud_pca[0, 0], aud_pca[0, 1], aud_pca[0, 2], c=color, marker='*', s=200)

        ax.set_xlabel('Componente Principal 1')
        ax.set_ylabel('Componente Principal 2')
        ax.set_zlabel('Componente Principal 3')
        plt.title('Características de Audios - PCA')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    # --------------------------------------------------------------------------

    # Métodos para el modo de registro y reporte de imágenes -------------------

    def tomar_foto(self):
        # Inicia la cámara 
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("No se pudo acceder a la cámara.")
        
        input("Presiona Enter para tomar la foto...")

        # Captura una imagen
        ret, frame = cap.read()

        if not ret:
            print(f"No se pudo capturar la imagen.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # Guarda la imagen en un archivo
        filename = self.temp_img_path
        cv2.imwrite(filename, frame)

        # Muestra la imagen capturada
        cv2.imshow(f'Foto Capturada', frame)
        cv2.waitKey(1000)  # Espera 1 segundo para mostrar la imagen

        # Cierra la ventana de visualización
        cv2.destroyAllWindows()

        # Libera la cámara
        cap.release()
        cv2.destroyAllWindows()

    def registrar_imagen(self):
        self.tomar_foto()
        img = Imagen(self.temp_img_path)
        self.clasificador_imagen.extract_features(img)
        self.clasificador_imagen.predict(img)

        return img, img.get_predicted_label()

    def agregar_imagen_log(self, img_obj):
        self.img_log.append(img_obj)

    def reporte_imagenes(self):
        if len(self.img_log) == 0:
            raise ValueError("Todavía no se han agregado imágenes al log.")

        total_imagenes = len(self.img_log)
        correct_predictions = sum(1 for img in self.img_log if img.real_label == img.predicted_label)

        et_true = [img.real_label for img in self.img_log]
        et_pred = [img.predicted_label for img in self.img_log]

        conf_matrix = confusion_matrix(et_true, et_pred)

        etiquetas = sorted(set(et_true + et_pred))

        return total_imagenes, correct_predictions, conf_matrix, etiquetas
    
    def heat_map_imagenes(self, conf_matrix, etiquetas):
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=etiquetas, yticklabels=etiquetas)
        plt.xlabel('Etiqueta predicha')
        plt.ylabel('Etiqueta real')
        plt.title('Matriz de Confusión')
        plt.show()
    
    def plot_session_imagenes(self):
        if len(self.img_log) == 0:
            raise ValueError("Todavía no se han agregado imágenes al log.")

        plt.figure(figsize=(10, 7))

        X_pca = self.clasificador_imagen.X_pca
        y = self.clasificador_imagen.y_dataset
        centroides = self.clasificador_imagen.centroides_pca
        etiquetas_centroides = self.clasificador_imagen.etiquetas_centroides

        colors = ['yellow', 'red', 'orange', 'green']
        labels = ['Banana', 'Manzana', 'Naranja', 'Pera']

        # Graficar los puntos del dataset
        for i, color in enumerate(colors):
            plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=color, label=labels[i])

        # Graficar los centroides
        for i, color in enumerate(colors):
            color = colors[etiquetas_centroides[i]]
            plt.scatter(centroides[i, 0], centroides[i, 1], c=color, marker='x', s=200)
            plt.text(centroides[i, 0], centroides[i, 1], f'Centroide {labels[etiquetas_centroides[i]]}', fontsize=9, ha='right', va='top')

        # Graficar las imágenes del log
        for img in self.img_log:
            img_pca = self.clasificador_imagen.pca.transform([img.get_features()])
            color = 'purple' if img.real_label == img.predicted_label else 'red'
            plt.scatter(img_pca[0, 0], img_pca[0, 1], c=color, marker='x', s=200)

        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('Características de Imágenes - PCA')
        plt.grid(True)
        plt.legend()
        plt.show()
        

    # --------------------------------------------------------------------------