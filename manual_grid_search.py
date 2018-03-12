# -*- coding: utf-8 -*-

import utils
import numpy as np
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
# PREPARANDO DATOS
# =============================================================================
clases = np.array(clases)
clases_entrenamiento = [clases.tolist().index(valor) for valor in clases_entrenamiento]
clases_test = [clases.tolist().index(clase) for clase in clases_test]

# =============================================================================
# PREPARANDO CONJUNTO DE PRUEBAS
# =============================================================================
normalizar = [True, False]
n_epochs = [5, 10, 25, 50, 100]
tasa_aprendizaje = [0.05, 0.1, 0.15, 0.2]
decrementar_tasa = [True, False]

mejor_rendimiento = 0.0
mejor_normalizar = True
mejor_n_epochs = 0.0
mejor_tasa_aprendizaje = 0.0
mejor_decrementar_tasa = True
combinaciones_probadas = 0

# Creamos las combinaciones con todas las posibilidades
for p_normalizar in normalizar:
    for p_n_epochs in n_epochs:
        for p_tasa_aprendizaje in tasa_aprendizaje:
            for p_decrementar_tasa in decrementar_tasa:

                # =============================================================================
                # INICIALIZANDO CLASIFICADOR
                # =============================================================================

                clasificador = ClasificadorPU(clases, norm=p_normalizar)

                # =============================================================================
                # ENTRENAMIENTO CLASIFICADORES
                # =============================================================================

                clasificador.entrena(entrenamiento, clases_entrenamiento, n_epochs=p_n_epochs,
                                     tasa_aprendizaje=p_tasa_aprendizaje, decrementar_tasa=p_decrementar_tasa)

                # =============================================================================
                # EVALUAMOS EL MODELO
                # =============================================================================

                rendimiento = clasificador.evalua_prob(conjunto_prueba=test, clases_conjunto_prueba=clases_test)
                evaluado = "Rendimiento Clasificador RECB Prob.: {0:.1f}%".format(round(rendimiento * 100, 1))
                print(evaluado)

                # Cogemos los mejores parámetros
                if rendimiento > mejor_rendimiento:
                    mejor_normalizar = p_normalizar
                    mejor_decrementar_tasa = p_decrementar_tasa
                    mejor_n_epochs = p_n_epochs
                    mejor_tasa_aprendizaje = p_tasa_aprendizaje
                    mejor_rendimiento = rendimiento

                # Aumentamos las combinaciones probadas
                combinaciones_probadas += 1

# =============================================================================
# RESULTADOS
# =============================================================================
resultado = "\nPara este clasificador, los mejores parámetros encontrados son: \n"
resultado += "\t Normalizar: {}\n".format(mejor_normalizar)
resultado += "\t Epochs: {}\n".format(mejor_n_epochs)
resultado += "\t Tasa aprendizaje: {}\n".format(mejor_tasa_aprendizaje)
resultado += "\t Decrementar tasa: {}\n".format(mejor_decrementar_tasa)
resultado += "\t Rendimiento: {0:.1f}%\n".format(round(mejor_rendimiento * 100, 1))
resultado += "\t Combinaciones probadas: {}\n".format(combinaciones_probadas)

print(resultado)

# =============================================================================
# FINAL - TIEMPOS DE EJECUCIÓN
# =============================================================================
utils.tiempo_ejecucion_obtenido(start_time)
