# -*- coding: utf-8 -*-

import utils
from copy import deepcopy
from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_error_cuadratico_estocastica import ClasificadorRECE
from clasificadores.regresion_verosimilitud_batch import ClasificadorRVB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
from datasets.votos import clases, entrenamiento, clases_entrenamiento, validacion, clases_validacion, test, \
    clases_test, ejemplo, ejemplo_clase

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
clasificadorRVB = ClasificadorRVB(clases)
clasificadorRVE = ClasificadorRVE(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

# Número de epochs a realizar
n_epochs = 10

# clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_recb = utils.recuperar_pesos(clasificadorRECB.fichero_de_volcado)
# clasificadorRECB.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_recb)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_rece = utils.recuperar_pesos(clasificadorRECE.fichero_de_volcado)
# clasificadorRECE.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_rece)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_rvb = utils.recuperar_pesos(clasificadorRVB.fichero_de_volcado)
# clasificadorRVB.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_rvb)

# Si existen pesos anteriores, los recuperará, si no serán 0.
pesos_iniciales_rve = utils.recuperar_pesos(clasificadorRVE.fichero_de_volcado)


# clasificadorRVE.entrena(entrenamiento, clases_entrenamiento, n_epochs, pesos_iniciales=pesos_iniciales_rve)

# ==================================================================================
# Conjunto aleatorio de elementos para probar el correcto funcionamiento del sistema
# ==================================================================================

ejemplos_independientes, clases_ejemplos_independientes = utils.generar_conjunto_independiente(len(ejemplo), 300, clasificadorRVE.pesos, clases)
ejemplos_dependientes = deepcopy(ejemplos_independientes)
clases_ejemplos_dependientes = utils.generar_conjunto_dependiente(clases_ejemplos_independientes, clases, 0.3)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

# =====================================================
# Regresión error cuadrático batch
# =====================================================
probabilidad = clasificadorRECB.clasifica_prob(ejemplo)
clase_probable = clasificadorRECB.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador RECB ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable,
                                                                                               porcentaje_exito)
print(clasificado)

# =====================================================
# Regresión error cuadrático estocástica
# =====================================================
probabilidad = clasificadorRECE.clasifica_prob(ejemplo)
clase_probable = clasificadorRECE.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador RECE ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable,
                                                                                               porcentaje_exito)
print(clasificado)

# =====================================================
# Regresión verosimilitud batch
# =====================================================
probabilidad = clasificadorRVB.clasifica_prob(ejemplo)
clase_probable = clasificadorRVB.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador RVB ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable,
                                                                                              porcentaje_exito)
print(clasificado)

# =====================================================
# Regresión verosimilitud estocástica
# =====================================================
probabilidad = clasificadorRVE.clasifica_prob(ejemplo)
clase_probable = clasificadorRVE.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador RVE ha clasificado a: '{0}' con una seguridad del {1:2f}%".format(clase_probable,
                                                                                              porcentaje_exito)
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

rendimiento = clasificadorRVB.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RVB Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

rendimiento = clasificadorRVE.evalua(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
evaluado = "Rendimiento Clasificador RVE Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
