# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils


class ClasificadorRVE(Clasificador):

    def __init__(self, clases: list, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/clasificador_rve"

        # Si tenemos pesos iniciales, los cargamos, si no, pesos es None
        self.pesos = utils.recuperar_pesos(self.fichero_de_volcado)

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None, decrementar_tasa=False):

        # Tenemos que añadir el término independiente a cada conjunto de datos
        entrenamiento = [[1] + elemento for elemento in entrenamiento]

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            self.pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            self.pesos = pesos_iniciales

        for _ in range(0, n_epochs):

            # Por cada conjunto del ejemplo
            for j, _ in enumerate(entrenamiento):

                # Por cada atributo del ejemplo
                for i, _ in enumerate(entrenamiento[j]):

                    # Comprobamos cual es nuestra clase objetivo y(j)
                    y = self.clases.index(clases_entrenamiento[j])

                    # Calculamos w*x también conocido como z
                    z = utils.pesos_por_atributo(self.pesos, entrenamiento[j])

                    # Sigma de z
                    sigma_z = utils.sigma(-z)

                    # Calculamos xi
                    xi = entrenamiento[j][i]

                    self.pesos[i] = self.pesos[i] + tasa_aprendizaje * (y - sigma_z) * xi

        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)

    def clasifica_prob(self, ejemplo):

        # Añadimos el término indendiente
        ejemplo = [1] + ejemplo

        return utils.sigma(-utils.pesos_por_atributo(self.pesos, ejemplo))

    def evalua(self, conjunto_prueba, clases_conjunto_prueba):

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
