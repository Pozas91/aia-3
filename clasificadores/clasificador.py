# -*- coding: utf-8 -*-


class Clasificador:

    def __init__(self, clases: list, norm=False):
        self.clases = clases
        self.norm = norm
        self.norm_media = 0.0
        self.norm_desviacion = 0.0

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None, decrementar_tasa=False):
        pass

    def clasifica_prob(self, ejemplo):
        pass

    def clasifica(self, ejemplo):
        pass

    def evalua(self, datos):
        pass

    def imprime(self):
        pass
