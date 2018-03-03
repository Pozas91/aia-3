# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils


class ClasificadorRVB(Clasificador):

    def __init__(self, clases: list, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/clasificador_rvb"

        # Si tenemos pesos iniciales, los cargamos, si no, pesos es None
        self.pesos = utils.recuperar_pesos(self.fichero_de_volcado)

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None,
                decrementar_tasa=False):

        # Guardamos la tasa de aprendizaje inicial
        tasa_aprendizaje_inicial = tasa_aprendizaje

        # Tenemos que añadir el término independiente a cada conjunto de datos
        entrenamiento = [[1] + elemento for elemento in entrenamiento]

        # Si se exige normalizar, normalizamos, si no, se mantiene tal y como viene.
        entrenamiento, self.norma = utils.normalizar_si_es_necesario(entrenamiento, self.normalizar, self.norma)

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            self.pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            self.pesos = pesos_iniciales

        for epoch in range(1, n_epochs + 1):

            # Por cada conjunto del ejemplo
            for j, _ in enumerate(entrenamiento):

                # Por cada atributo del ejemplo
                for i, _ in enumerate(entrenamiento[j]):

                    sumatorio = 0.0

                    # Por cada conjunto del ejemplo
                    for j1, _ in enumerate(entrenamiento):

                        # Comprobamos cual es nuestra clase objetivo y(j)
                        y = self.clases.index(clases_entrenamiento[j1])

                        # Sacamos wx(j)
                        z = utils.pesos_por_atributo(self.pesos, entrenamiento[j])

                        if z <= -700:
                            print(z)

                        # Sacamos sigma de wx(j)
                        sigma_z = utils.sigma(-z)

                        # Sacamos Xi(j)
                        xi = entrenamiento[j1][i]

                        sumatorio += (y - sigma_z) * xi

                    self.pesos[i] = self.pesos[i] + tasa_aprendizaje * sumatorio

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

        # Guardamos los pesos para reutilizarlos posteriormente
        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)

    def clasifica_prob(self, ejemplo):

        # Añadimos el término indendiente
        ejemplo = [1] + ejemplo

        # Si se exige normalización, normalizamos
        ejemplo, self.norma = utils.normalizar_fila_si_es_necesario(ejemplo, self.normalizar, self.norma)

        return utils.sigma(-utils.pesos_por_atributo(self.pesos, ejemplo))

    def evalua(self, conjunto_prueba, clases_conjunto_prueba):

        # Calcula ejemplos correctamente clasificados
        res = 0

        # Si se exige normalizar, normalizamos, si no, se mantiene tal y como viene.
        conjunto_prueba, self.norma = utils.normalizar_si_es_necesario(conjunto_prueba, self.normalizar, self.norma)

        # Por cada dato del conjunto de prueba
        for i, _ in enumerate(conjunto_prueba):
            # Sacamos la probabilidad de la clasificación
            probabilidad_clasificada = self.clasifica_prob(conjunto_prueba[i])

            # Asociamos esa probabilidad a una clase
            clase_clasificada = self.clases[round(probabilidad_clasificada)]

            res += clase_clasificada == clases_conjunto_prueba[i]

        rendimiento = res / len(conjunto_prueba)

        return rendimiento
