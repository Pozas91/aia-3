# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils


class ClasificadorPU(Clasificador):

    def __init__(self, clases: list, norm=False):
        Clasificador.__init__(self, clases, norm)

        # Ruta del fichero donde haremos el volcado de información
        self.fichero_de_volcado = "datasets/clasificador_pu"

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

        array_errores = []

        for epoch in range(1, n_epochs + 1):

            # o = umbral (w * x)
            # wi <- wi + tasa de aprendizaje * xi * (y - o)
            errores = 0

            for j, _ in enumerate(entrenamiento):
                o = utils.umbral(utils.pesos_por_atributo(self.pesos, entrenamiento[j]))
                y = utils.convierte_republicano_democrata(clases_entrenamiento[j])
                actualizacion = tasa_aprendizaje * (y - o)

                for wi, _ in enumerate(self.pesos):

                    if y != o:
                        self.pesos[wi] += entrenamiento[j][wi] * actualizacion

                    errores += int(actualizacion != 0.0)

            array_errores.append(errores)

            # Si está activada la opción de decrementar la tasa, la decrementamos
            if decrementar_tasa:
                tasa_aprendizaje = utils.rate_decay(tasa_aprendizaje_inicial, epoch)

        # Generamos el gráfico
        utils.generar_grafico(array_errores, 'Perceptrón Umbral')

        # Guardamos los pesos para reutilizarlos posteriormente
        utils.guardar_pesos(self.fichero_de_volcado, self.pesos)
