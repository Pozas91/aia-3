# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
from copy import deepcopy


class ClasificadorOVR:

    def __init__(self, clases: list, clasificador: Clasificador):
        self.clases = clases
        self.entrenamientos = []
        self.clasificador = clasificador
        self.clasificador_sin_entrenar = deepcopy(clasificador)

    def entrena(self, entrenamiento: list, clases_entrenamiento: list, n_epochs):
        for clase in self.clases:
            clases_entrenamiento_aux = [clase if valor == self.clases.tolist().index(clase) else '-' for valor in clases_entrenamiento]
            self.clasificador.clases = ['-', clase]
            self.clasificador.entrena(entrenamiento, clases_entrenamiento_aux, n_epochs)
            self.entrenamientos.append(self.clasificador)
            self.clasificador = deepcopy(self.clasificador_sin_entrenar)

    def clasifica(self, ejemplo) -> (float, str):

        max_prob = 0.0
        max_clases = ''

        for clasificador_entrenado in self.entrenamientos:

            prob = clasificador_entrenado.clasifica_prob(ejemplo)

            if prob > max_prob:
                max_prob = prob
                max_clases = clasificador_entrenado.clases[1]

        return max_prob, max_clases