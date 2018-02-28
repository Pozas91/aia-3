# -*- coding: utf-8 -*-

import utils
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.perceptron_umbral import ClasificadorPU
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, clases_test

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# CLASIFICADORES
# =============================================================================

clasificadorRECB = ClasificadorRECB(clases)
#clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, 10)


clasificadorPU = ClasificadorPU(clases)
clasificadorPU.entrena(entrenamiento, clases_entrenamiento, 10)


# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
