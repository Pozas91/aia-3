# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils
import math
import numpy as np


class ClasificadorRVB(Clasificador):

    def __init__(self, clases: np.ndarray, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/pesos/rvb"

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
            self.pesos = [random.uniform(-1, 1) for _ in range(len(entrenamiento[0]))]
        else:
            self.pesos = pesos_iniciales

        # Lista de los errores
        errores = []

        # para cada ejemplo del conjunto dividir restarle su media y dividirle la desviación tipica

        for epoch in range(1, n_epochs + 1):

            # Inicializamos las variables para calcular la tasa de error
            error_ejemplo_y_uno = 0.0
            error_ejemplo_y_cero = 0.0

            # Por cada atributo del ejemplo
            for i, _ in enumerate(entrenamiento[0]):

                sumatorio = 0.0

                # Por cada conjunto del ejemplo
                for j, _ in enumerate(entrenamiento):

                    # Comprobamos cual es nuestra clase objetivo y(j)
                    y = clases_entrenamiento[j]

                    # Sacamos wx(j)
                    z = utils.pesos_por_atributo(self.pesos, entrenamiento[j])

                    # Sacamos sigma de wx(j)
                    sigma = utils.sigmoide(-z)

                    # Sacamos Xi(j)
                    xi = entrenamiento[j][i]

                    sumatorio += (y - sigma) * xi

                    # Tasa error
                    # Notación: D+ son los ejemplos (x,y) de D con y = 1; D- son aquellos con y = 0
                    # Primer sumatorio corresponde a D+ y el segundo sumatorio a D-
                    # LL(w) = - Sumatorio (log (1 + e^-w*x)) - Sumatorio (log (1 + e^w*x))
                    if y == 1:
                        error_ejemplo_y_uno += math.log1p(math.exp(-z))
                    elif y == 0:
                        error_ejemplo_y_cero += math.log1p(math.exp(z))

                self.pesos[i] = self.pesos[i] + tasa_aprendizaje * sumatorio

            # Guardamos el error en la lista
            error = - error_ejemplo_y_uno - error_ejemplo_y_cero
            errores.append(error)

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

        # Generamos el gráfico
        utils.generar_grafico(errores, 'Regresión verosimilitud batch', y_label='Verosimilitud')

        # Guardamos los pesos para reutilizarlos posteriormente
        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)
