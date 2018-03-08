# -*- coding: utf-8 -*-

import utils


class Clasificador:

    def __init__(self, clases: list, normalizar=False):
        self.clases = clases
        self.normalizar = normalizar
        self.fichero_de_volcado = None
        self.pesos = []
        self.means = []
        self.std = []

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None,
                decrementar_tasa=False):
        pass

    """
    Este método clasifica el ejemplo dado basado en la función umbral
    """

    def clasifica(self, ejemplo):

        # Si se exige normalización, normalizamos
        ejemplo = utils.normalizar_fila_si_es_necesario(ejemplo, self.normalizar, self.means, self.std)

        # Añadimos el término indendiente
        ejemplo = [1] + ejemplo

        return self.clases[utils.umbral(utils.pesos_por_atributo(self.pesos, ejemplo))]

    """
    Este método clasifica el ejemplo dado basado en la función sigmoide
    """

    def clasifica_prob(self, ejemplo):

        # Si se exige normalización, normalizamos
        ejemplo = utils.normalizar_fila_si_es_necesario(ejemplo, self.normalizar, self.means, self.std)

        # Añadimos el término indendiente
        ejemplo = [1] + ejemplo

        return utils.sigmoide(-utils.pesos_por_atributo(self.pesos, ejemplo))

    """
    Este método evalua el rendimiento del clasificador basado en la función umbral
    """

    def evalua(self, conjunto_prueba, clases_conjunto_prueba):

        # Calcula ejemplos correctamente clasificados
        res = 0

        # Por cada dato del conjunto de prueba
        for i, _ in enumerate(conjunto_prueba):
            # Sacamos la clase clasificada
            clase_clasificada = self.clasifica(conjunto_prueba[i])

            res += clase_clasificada == clases_conjunto_prueba[i]

        rendimiento = res / len(conjunto_prueba)

        return rendimiento

    """
    Este método evalua el rendimiento del clasificador basado en la función sigmoide
    """

    def evalua_prob(self, conjunto_prueba, clases_conjunto_prueba):

        # Calcula ejemplos correctamente clasificados
        res = 0

        # Por cada dato del conjunto de prueba
        for i, _ in enumerate(conjunto_prueba):
            # Sacamos la probabilidad de la clasificación
            probabilidad_clasificada = self.clasifica_prob(conjunto_prueba[i])

            # Asociamos esa probabilidad a una clase
            clase_clasificada = self.clases[round(probabilidad_clasificada)]

            res += clase_clasificada == clases_conjunto_prueba[i]

        rendimiento = res / len(conjunto_prueba)

        return rendimiento

    """
    Imprime el clasificador
    """
    def imprime(self):
        return self.pesos
