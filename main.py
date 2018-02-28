# -*- coding: utf-8 -*-

import utils
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, clases_test, ejemplo, ejemplo_clase

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# INICIALIZACIÓN CLASIFICADORES
# =============================================================================

clasificadorRECB = ClasificadorRECB(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

pesos_iniciales_recb = utils.recuperar_pesos(clasificadorRECB.fichero_de_volcado)
n_epochs = 10
clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_recb)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

probabilidad = clasificadorRECB.clasifica_prob(ejemplo)

clasificado = "El Clasificador RECB Prob. ha clasificado el ejemplo con '{}' siendo el más probable '{}'".format(probabilidad, clasificadorRECB.clases[round(probabilidad)])
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

rendimiento = clasificadorRECB.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RECB Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
