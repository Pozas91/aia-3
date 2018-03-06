# -*- coding: utf-8 -*-

from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_error_cuadratico_estocastica import ClasificadorRECE
from clasificadores.regresion_verosimilitud_batch import ClasificadorRVB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
import random
import utils


class ClasificadorOVR:

    # =============================================================================
    # ENTRENAMIENTO CLASIFICADORES
    #
    # PU -> Perceptrón Umbral
    # RECB -> Regresión error cuadratico batch
    # RECE -> Regresión error cuadratico estocastica
    # RVB -> Regresión verosimilitud batch
    # RVE -> Regresión verosimilitud estocastica
    # 
    # =============================================================================

    @staticmethod
    def entrena(clases: list, entrenamiento: list, clases_entrenamiento: list, ejemplo: list, ejemplo_clases: list,
                clasificador: str, n_epochs):

        max_prob = 0
        max_clase = ''

        indice_aleatorio = random.randrange(len(ejemplo))
        ejemplo = ejemplo[indice_aleatorio]
        ejemplo_clases = ejemplo_clases[indice_aleatorio]
        # aux_clases = utils.genera_lista_one_vs_rest(clases, ejemplo_clases)

        for clase in clases:
            aux_clases = utils.genera_lista_one_vs_rest(clases, clase)
            print("Clases: {0}".format(clases))
            print("Aux clases: {0}".format(aux_clases))

            if clasificador == 'PU':

                clasificadorPU = ClasificadorPU(aux_clases)
                clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)
                clase_clasificada_prob = clasificadorPU.clasifica_prob(ejemplo)

            elif clasificador == 'RECB':

                clasificadorRECB = ClasificadorRECB(aux_clases)
                # Si existen pesos anteriores, los recuperará, si no serán 0.
                pesos_iniciales_recb = utils.recuperar_pesos(clasificadorRECB.fichero_de_volcado)
                clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, n_epochs,
                                         pesos_iniciales=pesos_iniciales_recb)
                clase_clasificada_prob = clasificadorRECB.clasifica_prob(ejemplo)

            elif clasificador == 'RECE':

                clasificadorRECE = ClasificadorRECE(aux_clases)
                # Si existen pesos anteriores, los recuperará, si no serán 0.
                pesos_iniciales_rece = utils.recuperar_pesos(clasificadorRECE.fichero_de_volcado)
                clasificadorRECE.entrena(entrenamiento, clases_entrenamiento, n_epochs,
                                         pesos_iniciales=pesos_iniciales_rece)
                clase_clasificada_prob = clasificadorRECE.clasifica_prob(ejemplo)

            elif clasificador == 'RVB':

                clasificadorRVB = ClasificadorRVB(aux_clases)
                # Si existen pesos anteriores, los recuperará, si no serán 0.
                pesos_iniciales_rvb = utils.recuperar_pesos(clasificadorRVB.fichero_de_volcado)
                clasificadorRVB.entrena(entrenamiento, clases_entrenamiento, n_epochs,
                                        pesos_iniciales=pesos_iniciales_rvb)
                clase_clasificada_prob = clasificadorRVB.clasifica_prob(ejemplo)

            else:
                clasificadorRVE = ClasificadorRVE(aux_clases)
                # Si existen pesos anteriores, los recuperará, si no serán 0.
                pesos_iniciales_rve = utils.recuperar_pesos(clasificadorRVE.fichero_de_volcado)
                clasificadorRVE.entrena(entrenamiento, clases_entrenamiento, n_epochs,
                                        pesos_iniciales=pesos_iniciales_rve)
                clase_clasificada_prob = clasificadorRVE.clasifica_prob(ejemplo)

            if clase_clasificada_prob > max_prob:
                max_prob = clase_clasificada_prob
                max_clase = clase

        print("Clase de ejemplo seleccionada aleatoriamente: {0}".format(ejemplo_clases))
        print("Clasificador " + clasificador + " ha clasificado a: '{0}'".format(max_clase))
