# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils


class ClasificadorRECB(Clasificador):

    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None, decrementar_tasa=False):

        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            pesos = pesos_iniciales

        epoch = 0

        while epoch < n_epochs:

            for j, _ in enumerate(entrenamiento):
                o = utils.sigma(utils.pesos_por_atributo(pesos, entrenamiento[j]))
                print(self.clases)
                print(o)
                pass

            epoch += 1

    def clasifica_prob(self, ejemplo):
        pass

    def clasifica(self, ejemplo):
        pass

    def evalua(self, datos):
        pass