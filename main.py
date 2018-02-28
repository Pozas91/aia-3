# -*- coding: utf-8 -*-

import utils
from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_error_cuadratico_estocastica import ClasificadorRECE
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, clases_test, ejemplo, ejemplo_clase

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# INICIALIZANDO CLASIFICADORES
# =============================================================================

clasificadorPU = ClasificadorPU(clases)
clasificadorRECB = ClasificadorRECB(clases)
clasificadorRECE = ClasificadorRECE(clases)
clasificadorRVE = ClasificadorRVE(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

n_epochs = 10
clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_recb = utils.recuperar_pesos(clasificadorRECB.fichero_de_volcado)
n_epochs = 10
# clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_recb)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_rece = utils.recuperar_pesos(clasificadorRECE.fichero_de_volcado)
n_epochs = 10
# clasificadorRECE.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_rece)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_rve = utils.recuperar_pesos(clasificadorRVE.fichero_de_volcado)
n_epochs = 10
clasificadorRVE.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_rve)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

probabilidad = clasificadorRECB.clasifica_prob(ejemplo)
clase_probable = clasificadorRECB.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)

clasificado = "Clasificador RECB ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable, porcentaje_exito)
print(clasificado)

probabilidad = clasificadorRECE.clasifica_prob(ejemplo)
clase_probable = clasificadorRECE.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)

clasificado = "Clasificador RECE ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable, porcentaje_exito)
print(clasificado)

probabilidad = clasificadorRVE.clasifica_prob(ejemplo)
clase_probable = clasificadorRVE.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)

clasificado = "Clasificador RVE ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable, porcentaje_exito)
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

rendimiento = clasificadorRECB.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RECB Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

rendimiento = clasificadorRECE.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RECE Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

rendimiento = clasificadorRVE.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RVE Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
