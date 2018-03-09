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
            self.clasificador.clases = ['-', clase]
            clases_entrenamiento_aux = [1 if valor == self.clases.tolist().index(clase) else 0 for valor in
                                        clases_entrenamiento]
            self.clasificador.entrena(entrenamiento, clases_entrenamiento_aux, n_epochs)
            self.entrenamientos.append(self.clasificador)
            self.clasificador = deepcopy(self.clasificador_sin_entrenar)

    def clasifica(self, ejemplo) -> (float, str):

        max_prob = 0.0
        max_class_index = 0

        for clasificador_entrenado in self.entrenamientos:

            prob = clasificador_entrenado.clasifica_prob(ejemplo)

            if prob > max_prob:
                max_prob = prob
                max_class_index = self.clases.tolist().index(clasificador_entrenado.clases[1])

        return max_prob, max_class_index

    def evalua(self, conjunto_prueba, clases_conjunto_prueba) -> float:

        # Calcula ejemplos correctamente clasificados
        res = 0

        # Por cada dato del conjunto de prueba
        for i, _ in enumerate(conjunto_prueba):

            # Sacamos la clase clasificada
            _, max_class_index = self.clasifica(conjunto_prueba[i])

            res += max_class_index == clases_conjunto_prueba[i]

        rendimiento = res / len(conjunto_prueba)

        return rendimiento
