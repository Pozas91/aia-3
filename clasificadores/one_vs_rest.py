# -*- coding: utf-8 -*-

from clasificadores.perceptron_umbral import ClasificadorPU
from clasificadores.regresion_error_cuadratico_batch import ClasificadorRECB
from clasificadores.regresion_error_cuadratico_estocastica import ClasificadorRECE
from clasificadores.regresion_verosimilitud_batch import ClasificadorRVB
from clasificadores.regresion_verosimilitud_estocastica import ClasificadorRVE

class ClasificadorOVR():

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
    
    @staticmethod
    def entrena(clases: list, entrenamiento: list, clases_entrenamiento: list, ejemplo: list, clasificador: str, n_epochs):
        
        for clase in clases:
            if clasificador == 'PU':
                clasificadorPU = ClasificadorPU(clase)
                clasificadorPU.entrena(entrenamiento, clases_entrenamiento, n_epochs)                
                clase_clasificada = clasificadorPU.clasifica(ejemplo)
                print(clase_clasificada)
                clasificado = "Clasificador PU ha clasificado a: '{0}'".format(clase_clasificada)
                print(clasificado)
            elif clasificador == 'RECB':
                pass
            elif clasificador == 'RECE':
                pass
            elif clasificador == 'RVB':
                pass
            else:
                pass

        
