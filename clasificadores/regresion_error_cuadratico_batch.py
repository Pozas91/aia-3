# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils
import math


class ClasificadorRECB(Clasificador):

    def __init__(self, clases: list, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/pesos/recb"

        # Si tenemos pesos iniciales, los cargamos, si no, pesos es None
        self.pesos = utils.recuperar_pesos(self.fichero_de_volcado)

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None,
                decrementar_tasa=False):

        # Guardamos la tasa de aprendizaje inicial
        tasa_aprendizaje_inicial = tasa_aprendizaje

        # Tenemos que añadir el término independiente a cada conjunto de datos
        entrenamiento = [[1] + elemento for elemento in entrenamiento]

        # Si se exige normalizar, normalizamos, si no, se mantiene tal y como viene.
        entrenamiento, self.means, self.std = utils.normalizar_si_es_necesario(entrenamiento, self.normalizar)

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            self.pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            self.pesos = pesos_iniciales

        # Lista de los errores
        errores = []

        for epoch in range(1, n_epochs + 1):

            # Inicializamos la variable error
            error = 0.0

            # Por cada atributo del ejemplo
            for i, _ in enumerate(entrenamiento[0]):

                sumatorio = 0.0

                # Por cada conjunto del ejemplo
                for j, _ in enumerate(entrenamiento):

                    # Comprobamos cual es nuestra clase objetivo y(j)
                    y = self.clases.index(clases_entrenamiento[j])

                    # Sacamos el vector de pesos por x(j)
                    z = utils.pesos_por_atributo(self.pesos, entrenamiento[j])

                    # Calculamos la o(j) mediante la función de sigma
                    o = utils.sigmoide(-z)

                    sumatorio += entrenamiento[j][i] * (y - o) * o * (1 - o)

                    # Sumatorio cuadrático medio
                    error += math.pow((y - o), 2)

                self.pesos[i] = self.pesos[i] + tasa_aprendizaje * sumatorio

            # Guardamos el error en la lista
            errores.append(error)

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

        # Generamos el gráfico
        utils.generar_grafico(errores, 'Regresión error cuadrático batch')

        # Guardamos los pesos para reutilizarlos posteriormente
        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)
