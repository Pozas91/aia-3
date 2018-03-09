# -*- coding: utf-8 -*-

import utils
import random
from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
# from datasets.digitdata import classes, training_data, training_classes, validation_data, validation_classes, test_classes, test_data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# PREPARANDO DATOS
# =============================================================================
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_names, y_names = iris.feature_names, iris.target_names
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.25)

# =============================================================================
# INICIALIZANDO CLASIFICADOR
# =============================================================================

clases = ['-', 'setosa']
clasificador = ClasificadorRECB(clases)

# =============================================================================
# ENTRENAMIENTO CLASIFICADORES
# =============================================================================

# Número de epochs a realizar
n_epochs = 100
tasa_aprendizaje = 0.1
decrementar_tasa = False

y_train_aux = [1 if data == 0 else 0 for data in y_train]
clasificador.entrena(X_train, y_train_aux, n_epochs)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

indice_aleatorio = random.randrange(len(X_train))
ejemplo = X_train[indice_aleatorio]
ejemplo_clases = y_names[y_train[indice_aleatorio]]

probabilidad = clasificador.clasifica_prob(ejemplo)
clase_probable = clasificador.clases[round(probabilidad)]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador PU ha clasificado a: '{0}' con una seguridad del {1:2f}% cuando el resultado era '{2}'".format(clase_probable,
                                                                                               porcentaje_exito, ejemplo_clases)
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

y_test = [1 if data == 0 else 0 for data in y_test]
rendimiento = clasificador.evalua(conjunto_prueba=X_test, clases_conjunto_prueba=y_test)
evaluado = "Rendimiento Clasificador PU: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
