# -*- coding: utf-8 -*-

import utils
import numpy as np
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
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

clasificadorRECB = ClasificadorRECB(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

# Número de epochs a realizar
n_epochs = 10
tasa_aprendizaje = 0.1
decrementar_tasa = False

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_recb = utils.recuperar_pesos(clasificadorRECB.fichero_de_volcado)
clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_recb)
clasificadorRECB.mostrar_grafico('Regresión error cuadrático batch')
clasificadorRECB.guardar_pesos()

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

probabilidad = clasificadorRECB.clasifica_prob(ejemplo)
clase_probable = clasificadorRECB.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador RECB ha clasificado a: '{0}' con una seguridad del {1:2f}% cuando debería ser '{2}'".format(
    clase_probable, porcentaje_exito, ejemplo_clase)
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

rendimiento = clasificadorRECB.evalua_prob(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RECB Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
