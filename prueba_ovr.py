# -*- coding: utf-8 -*-

import utils
from clasificadores.one_vs_rest import ClasificadorOVR
from clasificadores.perceptron_umbral import ClasificadorPU
from datasets.digitdata import classes, training_data, training_classes, validation_data, validation_classes, test_classes, \
    test_data

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

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
clasificadorPU = ClasificadorPU(None)
clasificadorOVR = ClasificadorOVR(classes, clasificadorPU)
clasificadorOVR.entrena(training_data, training_classes, 1)
clasificadorOVR.clasifica(test_data[0])

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
