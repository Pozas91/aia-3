# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import numpy as np
import utils
import numpy as np


class ClasificadorPU(Clasificador):

    def __init__(self, clases: np.ndarray, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/pesos/pu"

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None,
                decrementar_tasa=False):

        # Guardamos la tasa de aprendizaje inicial
        tasa_aprendizaje_inicial = tasa_aprendizaje

        # Si se exige normalizar, normalizamos, si no, se mantiene tal y como viene.
        entrenamiento, self.means, self.std = utils.normalizar_si_es_necesario(entrenamiento, self.normalizar)

        # Tenemos que añadir el término independiente a cada conjunto de datos
        entrenamiento = np.insert(entrenamiento, 0, 1, axis=1)

        # Numero de epochs (veces que se itera sobre el conjunto completo de datos)

        # Rate decay: booleano que indica si la tasa de aprendizaje debe ir
        # disminuyendo en función del número de actualizaciones realizadas

        # Pesos iniciales: si es None los pesos iniciales son aleatorios (por
        # ejemplo, entre -1 y 1). Si no es None, se proporciona la lista de pesos iniciales

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            self.pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            self.pesos = pesos_iniciales

        errores = []

        for epoch in range(1, n_epochs + 1):

            # o = umbral (w * x)
            # wi <- wi + tasa de aprendizaje * xi * (y - o)
            error = 0

            # Ordena aleatoriamente los indices
            random_indices = utils.random_indices(len(entrenamiento))

            for i in random_indices:

                # Multiplicamos los pesos por los atributos
                wx = utils.pesos_por_atributo(self.pesos, entrenamiento[i])

                # Sacamos nuestra clase actual
                o = utils.umbral(wx)

                # Sacamos cual es nuestra clase objetivo
                y = clases_entrenamiento[i]

                actualizacion = tasa_aprendizaje * (y - o)

                for j, _ in enumerate(entrenamiento[i]):

                    # Actualizamos el pesos
                    self.pesos[j] += entrenamiento[i][j] * actualizacion

                    # Almacenamos el error
                    error += int(actualizacion != 0.0)

            errores.append(error)

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

        # Generamos el gráfico
        utils.generar_grafico(errores, 'Perceptrón Umbral')

        # Guardamos los pesos para reutilizarlos posteriormente
        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)
