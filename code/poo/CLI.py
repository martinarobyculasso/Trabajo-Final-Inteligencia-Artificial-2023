import cmd
from colorama import init, Fore, Style

from Controlador import Controlador

init(autoreset=True)


class CLI(cmd.Cmd):
    intro = '\n'
    intro += Fore.WHITE + '=' * 100 + '\n'
    intro += Fore.MAGENTA + '¡Hola! Esta es una máquina expendedora de frutas activada por voz.\n\n'
    intro += Fore.WHITE + 'Escriba "help" para ver los comandos disponibles. Para salir, escriba "exit".\n'
    intro += '=' * 100 + '\n'
    prompt = Fore.CYAN + '>> '

    def __init__(self):
        super().__init__()
        self.controlador = Controlador()

    def do_tomar_fotos(self, arg):
        """Toma fotos de las frutas en las estanterías. Uso: tomar_fotos"""
        try:
            print('\n' + Fore.WHITE + 'Tomando fotos de las frutas...' + '\n')
            self.controlador.tomar_fotos()
            print(Fore.GREEN + f'\nImágenes tomadas y almacenadas en {self.controlador.img_path}.\n')
        except Exception as e:
            print(Fore.RED + f'Error al tomar fotos: {e}')

    def do_grabar_audio(self, arg):
        """Graba un audio y lo almacena para ser clasificado. Uso: grabar_audio"""
        while True:
            try:
                print(
                    '\n' + Fore.WHITE + 'Presiona Enter para comenzar a grabar. La grabación se detendrá automáticamente después de 3 segundos.')
                input(Fore.WHITE + 'Presiona Enter para comenzar...')

                self.controlador.grabar_audio()
                print(Fore.GREEN + f'\nAudio grabado y almacenado en {self.controlador.audio_path}.\n')

                response = input(Fore.WHITE + '¿Desea grabar otro audio? (s/n): ').strip().lower()
                if response == 'n':
                    print('\n')
                    break
                elif response != 's':
                    print(Fore.RED + '\n' +'Respuesta no válida. Por favor, responda con "s" o "n".')
            except Exception as e:
                print(Fore.RED + f'Error al grabar audio: {e}')
                break

    def do_obtener_fruta(self, arg):
        """Una vez que se ha grabado un audio y se han tomado las fotos, se clasifica la fruta. Uso: obtener_fruta"""
        try:
            print('\n' + Fore.WHITE + 'Clasificando audio...' + '\n')
            fruta = self.controlador.clasificar_audio()
            print(Fore.YELLOW + f'La fruta grabada es una {fruta}.\n')

            correct = input(Fore.WHITE + '¿La predicción fue correcta? (s/n): ').strip().lower()
            if correct == 's':
                print('\n' + Fore.WHITE + 'Clasificando imágenes...' + '\n')
                e1, e2, e3, e4 = self.controlador.clasificar_imagenes_estantes()
                print(Fore.WHITE + 'Estante 1: ', e1)
                print(Fore.WHITE + 'Estante 2: ', e2)
                print(Fore.WHITE + 'Estante 3: ', e3)
                print(Fore.WHITE + 'Estante 4: ', e4)
                try:
                    estante = self.controlador.obtener_estante()
                    if estante is not None:
                        print(Fore.YELLOW + '\n' + f'La fruta se encuentra en el estante {estante}.\n')
                        self.controlador.conectar()
                        self.controlador.send(fruta, estante)
                        self.controlador.desconectar()
                    else:
                        print(Fore.RED + '\n' + 'No se encontró la fruta pedida en ningún estante.\n')
                except ValueError as e:
                    print(Fore.RED + '\n' + f'Error: {e}\n')
                self.controlador.mostrar_estantes(estante)
            else:
                print(Fore.RED + '\n' + 'Por favor, vuelva a grabar el audio y tomar las fotos de las frutas.')

        except Exception as e:
            print(Fore.RED + '\n' + f'Error al obtener fruta: {e}' + '\n')

    def do_graficar_audio(self, arg):
        """Grafica las características del audio clasificado junto con las del dataset. Uso: graficar_audio"""
        try:
            print('\n' + Fore.WHITE + 'Graficando...' + '\n')
            self.controlador.plot_audio()
        except Exception as e:
            print(Fore.RED + f'Error al graficar audio: {e}' + '\n')

    def do_graficar_imagenes(self, arg):
        """Grafica las características de las imágenes clasificadas junto con las del dataset. Uso: plot_img"""
        try:
            print('\n' + Fore.WHITE + 'Graficando...' + '\n')
            self.controlador.plot_images()
        except Exception as e:
            print(Fore.RED + f'Error al graficar imágenes: {e}')

    def do_registrar_audios(self, arg):
        """Graba y clasifica múltiples audios, almacenándolos en un log. Uso: registrar_audios"""
        while True:
            try:
                print(
                    '\n' + Fore.WHITE + 'Presiona Enter para comenzar a grabar. La grabación se detendrá automáticamente después de 3 segundos.')
                input(Fore.WHITE + 'Presiona Enter para comenzar...')

                predicted_label = self.controlador.registrar_audio()
                print(Fore.GREEN + f'\nAudio grabado correctamente.\n')
                print(Fore.YELLOW + f'Predicción: {predicted_label}\n')

                correct = input(Fore.WHITE + '¿La predicción fue correcta? (s/n): ').strip().lower()
                if correct != 's':
                    real_label = input(Fore.WHITE + 'Ingrese la etiqueta correcta: ').strip().lower()
                    self.controlador.clasificador_audio.audio_obj.set_real_label(real_label)
                else:
                    self.controlador.clasificador_audio.audio_obj.set_real_label(predicted_label)

                self.controlador.agregar_audio_log()

                another = input(Fore.WHITE + '\n¿Desea grabar otro audio? (s/n): ').strip().lower()
                if another != 's':
                    break
            except Exception as e:
                print(Fore.RED + '\n' + f'Error al registrar audios: {e}')

        generar_reporte = input(Fore.WHITE + '\n¿Desea generar un reporte de la sesión? (s/n): ').strip().lower()
        if generar_reporte == 's':
            try:
                total_audios, correct_predictions, conf_matrix, etiquetas = self.controlador.reporte_audios()
                print(Fore.WHITE + '\n' + '=' * 100)
                print(Fore.WHITE + f'Cantidad de audios registrados: {total_audios}')
                print(Fore.WHITE + f'Cantidad de audios predichos correctamente: {correct_predictions}')
                print(Fore.WHITE + '=' * 100 + '\n')
                self.controlador.heat_map_audios(conf_matrix, etiquetas)
                self.controlador.plot_session()
            except Exception as e:
                print(Fore.RED + f'Error al generar el reporte: {e}')

    def do_registrar_imagenes(self, arg):
        """Captura y clasifica múltiples imagenes, almacenándolas en un log. Uso: registrar_imagenes"""
        while True:
            try:
                print(Fore.WHITE + '\n' + 'Esperando a que se abra la cámara...')
                img_temp, predicted_label_img = self.controlador.registrar_imagen()
                print(Fore.GREEN + f'\nImagen capturada correctamente.\n')
                print(Fore.YELLOW + f'Predicción: {predicted_label_img}\n')

                correct = input(Fore.WHITE + '¿La predicción fue correcta? (s/n): ').strip().lower()
                if correct != 's':
                    real_label = input(Fore.WHITE + 'Ingrese la etiqueta correcta: ').strip().lower()
                    img_temp.set_real_label(real_label)
                else:
                    img_temp.set_real_label(predicted_label_img)

                self.controlador.agregar_imagen_log(img_temp)

                another = input(Fore.WHITE + '\n¿Desea tomar más fotos? (s/n): ').strip().lower()
                if another != 's':
                    break
            except Exception as e:
                print(Fore.RED + '\n' + f'Error al registrar imágenes: {e}')

        generar_reporte = input(Fore.WHITE + '\n¿Desea generar un reporte de la sesión? (s/n): ').strip().lower()
        if generar_reporte == 's':
            try:
                total_imagenes, correct_predictions, conf_matrix, etiquetas = self.controlador.reporte_imagenes()
                print(Fore.WHITE + '\n' + '=' * 100)
                print(Fore.WHITE + f'Cantidad de imágenes registradas: {total_imagenes}')
                print(Fore.WHITE + f'Cantidad de imágenes predichas correctamente: {correct_predictions}')
                print(Fore.WHITE + '=' * 100 + '\n')
                self.controlador.heat_map_imagenes(conf_matrix, etiquetas)
                self.controlador.plot_session_imagenes()
            except Exception as e:
                print(Fore.RED + f'Error al generar el reporte: {e}')

    def do_exit(self, arg):
        """Sale del programa. Uso: exit"""
        print('\n' + Fore.WHITE + 'Terminando programa...' + '\n')
        return True

    def do_help(self, arg):
        if arg:
            try:
                func = getattr(self, 'do_' + arg)
                print('\n' + Fore.WHITE + func.__doc__ + '\n')
            except AttributeError:
                print(Fore.RED + f"\nNo hay ayuda disponible para el comando '{arg}'\n")
        else:
            print('\n' + Fore.WHITE + "Comandos disponibles:")
            print(Fore.WHITE + '=' * 100 + '\n')
            print(Fore.WHITE + "tomar_fotos          " + Fore.WHITE + "Toma fotos de las frutas en las estanterías.")
            print(Fore.WHITE + "grabar_audio         " + Fore.WHITE + "Graba un audio para ser clasificado.")
            print(Fore.WHITE + "obtener_fruta        " + Fore.WHITE + "Clasifica el audio y las imágenes y obtiene el estante correspondiente.")
            print(Fore.WHITE + "graficar_audio       " + Fore.WHITE + "Grafica las características del audio clasificado.")
            print(Fore.WHITE + "graficar_imagenes    " + Fore.WHITE + "Grafica las características de las imágenes clasificadas.")
            print(Fore.WHITE + "registrar_audios     " + Fore.WHITE + "Graba y clasifica múltiples audios, almacenándolos en un log.")
            print(Fore.WHITE + "registrar_imagenes   " + Fore.WHITE + "Captura y clasifica múltiples imagenes, almacenándolas en un log.")
            print(Fore.WHITE + "exit                 " + Fore.WHITE + "Sale del programa." + '\n')
            print(Fore.WHITE + '=' * 100 + '\n')

    def default(self, line):
        print(Fore.RED + f"\nComando no reconocido: {line}\n")

    # Deshabilitar modo registro para el examen
