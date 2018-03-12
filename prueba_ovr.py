# -*- coding: utf-8 -*-

import utils
import random
import numpy as np
from clasificadores.one_vs_rest import ClasificadorOVR
from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE
from datasets.digitdata import classes, training_data, training_classes, validation_data, validation_classes, \
    test_classes, test_data

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split

# =============================================================================
# COMIENZO - TIEMPOS DE EJECUCIÓN
# =============================================================================
start_time = utils.comienzo_tiempo_ejecucion()

# =============================================================================
# PREPARANDO DATOS
# =============================================================================
# iris = load_iris()
# X_iris, y_iris = iris.data, iris.target
# X_names, y_names = iris.feature_names, iris.target_names
# X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.25)

classes = np.array(classes)
y_names = classes
X_train = training_data
y_train = [classes.tolist().index(clase) for clase in training_classes]
X_test = test_data
y_test = [classes.tolist().index(clase) for clase in test_classes]

# =============================================================================
# INICIALIZANDO CLASIFICADOR
# =============================================================================

n_epochs = 10
clasificador = ClasificadorRVE(None)

clasificadorOVR = ClasificadorOVR(y_names, clasificador)
clasificadorOVR.entrena(X_train, y_train, n_epochs)

# =============================================================================
# CLASIFICACION DE EJEMPLOS
# =============================================================================

indice_aleatorio = random.randrange(len(X_train))
ejemplo = X_train[indice_aleatorio]
ejemplo_clases = y_names[y_train[indice_aleatorio]]

probabilidad, clase_index = clasificadorOVR.clasifica(ejemplo)
clase_probable = clasificadorOVR.clases[clase_index]
porcentaje_exito = utils.ponderar_probabilidad(probabilidad)
clasificado = "Clasificador OVR ha clasificado a: '{0}' con una seguridad del {1:2f}% cuando el resultado era '{2}'".format(
    clase_probable,
    porcentaje_exito, ejemplo_clases)
print(clasificado)

# =============================================================================
# EVALUAMOS EL MODELO
# =============================================================================

rendimiento = clasificadorOVR.evalua(conjunto_prueba=X_test, clases_conjunto_prueba=y_test)
evaluado = "Rendimiento Clasificador OVR: {0:.1f}%".format(round(rendimiento * 100, 1))
print(evaluado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
