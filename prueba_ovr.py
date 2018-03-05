# -*- coding: utf-8 -*-

import utils
from clasificadores.one_vs_rest import ClasificadorOVR
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, \
    clases_test, ejemplo, ejemplo_clase

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
clasificadorOVR = ClasificadorOVR.entrena(clases, entrenamiento, clases_entrenamiento, ejemplo, 'PU', 100)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
