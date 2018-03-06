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
            
            clases_entrenamiento_aux = ['1' if valor == clase else '0' for valor in clases_entrenamiento]
            self.clasificador.clases = ['-', clase]
            self.clasificador.entrena(entrenamiento, clases_entrenamiento_aux, n_epochs)
            self.entrenamientos.append(self.clasificador)
            self.clasificador = deepcopy(self.clasificador_sin_entrenar)
