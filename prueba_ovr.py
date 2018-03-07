# -*- coding: utf-8 -*-

import utils
import random
from clasificadores.one_vs_rest import ClasificadorOVR
from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
# from datasets.digitdata import classes, training_data, training_classes, validation_data, validation_classes, test_classes, test_data
from sklearn.datasets import load_iris

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
iris = load_iris()
clasificador = ClasificadorPU([])
clasificadorOVR = ClasificadorOVR(iris.target_names, clasificador)
clasificadorOVR.entrena(iris.data, iris.target, 10)

indice_aleatorio = random.randrange(len(iris.data))
ejemplo = iris.data[indice_aleatorio]
ejemplo_clases = iris.target[indice_aleatorio]

clasificado = clasificadorOVR.clasifica(ejemplo)
print(clasificado)
print(iris.target_names[ejemplo_clases])

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
