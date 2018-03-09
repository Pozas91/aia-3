# -*- coding: utf-8 -*-

import utils
import numpy as np
from clasificadores.perceptron_umbral import ClasificadorPU
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, \
    clases_test, ejemplo, ejemplo_clase

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# PREPARANDO DATOS
# =============================================================================
clases = np.array(clases)
clases_entrenamiento = [clases.tolist().index(valor) for valor in clases_entrenamiento]
clases_test = [clases.tolist().index(clase) for clase in clases_test]

# =============================================================================
# INICIALIZANDO CLASIFICADOR
# =============================================================================

clasificadorPU = ClasificadorPU(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

# Número de epochs a realizar
n_epochs = 100
tasa_aprendizaje = 0.1
decrementar_tasa = False

# Si existen pesos anteriores, los recuperará, si no serán 0.
clasificadorPU.cargar_pesos_guardados()
clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

clase_index = clasificadorPU.clasifica(ejemplo)
clase_probable = clasificadorPU.clases[clase_index]
clasificado = "Clasificador PU ha clasificado a: '{0}' cuando debería ser '{1}'".format(clase_probable, ejemplo_clase)
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

rendimiento = clasificadorPU.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador PU: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
