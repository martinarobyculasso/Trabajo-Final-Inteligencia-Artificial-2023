import cv2


class Imagen:
    def __init__(self, path):
        self.path = path
        self.features = None
        self.predicted_label = None
        self.real_label = None

    def set_features(self, features):
        self.features = features

    def get_features(self):
        return self.features

    def set_predicted_label(self, predicted_label):
        self.predicted_label = predicted_label

    def get_predicted_label(self):
        return self.predicted_label

    def set_real_label(self, real_label):
        self.real_label = real_label

    def get_real_label(self):
        return self.real_label

    # Nota: No implementé método get para path, se accede a este atributo directamente