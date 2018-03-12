# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils
import math
import numpy as np
import os


class ClasificadorRECE(Clasificador):

    def __init__(self, clases: np.ndarray, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/pesos/rece')

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None,
                decrementar_tasa=False):

        # Guardamos la tasa de aprendizaje inicial
        tasa_aprendizaje_inicial = tasa_aprendizaje

        # Si se exige normalizar, normalizamos, si no, se mantiene tal y como viene.
        entrenamiento, self.means, self.std = utils.normalizar_si_es_necesario(entrenamiento, self.normalizar)

        # Tenemos que añadir el término independiente a cada conjunto de datos
        entrenamiento = np.insert(entrenamiento, 0, 1, axis=1)

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            self.pesos = utils.generar_pesos_aleatorios(len(entrenamiento[0]))
        else:
            self.pesos = pesos_iniciales

        for epoch in range(1, n_epochs + 1):

            # Inicializamos la variable error
            error = 0.0

            # Ordena aleatoriamente los indices
            random_indices = utils.random_indices(len(entrenamiento))

            # Por cada conjunto del ejemplo
            for i in random_indices:

                # Por cada atributo del ejemplo
                for j, _ in enumerate(entrenamiento[i]):

                    # Comprobamos cual es nuestra clase objetivo y(j)
                    y = clases_entrenamiento[i]

                    # Sacamos el vector de pesos por x(j)
                    z = utils.pesos_por_atributo(self.pesos, entrenamiento[i])

                    # Calculamos la o(j) mediante la función de sigma
                    o = utils.sigmoide(-z)

                    # Sumatorio cuadrático medio
                    error += math.pow((y - o), 2)

                    self.pesos[j] = self.pesos[j] + tasa_aprendizaje * entrenamiento[i][j] * (y - o) * o * (1 - o)

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

            # Guardamos el error en la lista
            self.errores.append(error)
