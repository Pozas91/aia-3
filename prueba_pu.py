# -*- coding: utf-8 -*-

import utils
from clasificadores.perceptron_umbral import ClasificadorPU
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, \
    clases_test, ejemplo, ejemplo_clase

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

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
pesos_iniciales_pu = utils.recuperar_pesos(clasificadorPU.fichero_de_volcado)
clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

clase_clasificada = clasificadorPU.clasifica(ejemplo)
clasificado = "Clasificador PU ha clasificado a: '{0}'".format(clase_clasificada)
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
