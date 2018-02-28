# -*- coding: utf-8 -*-

from clasificadores.clasificador import Clasificador
import random
import utils


class ClasificadorPU(Clasificador):
    
    def entrena(self, entrenamiento, clases_entrenamiento, n_epochs, tasa_aprendizaje=0.1, pesos_iniciales=None, decrementar_tasa=False):
        
        # Numero de epochs (veces que se itera sobre el conjunto completo de datos)

        # Rate decay: booleano que indica si la tasa de aprendizaje debe ir
        # disminuyendo en funcion del numero de actualizaciones realizadas
        
        # Pesos iniciales: si es None los pesos iniciales son aleatorios (por
        # ejemplo, entre -1 y 1). Si no es None, se proporciona la lista de pesos iniciales
        
        # Si los pesos iniciales son None, entonces los iniciaremos aleatoriamente con un número de entre -1 y 1
        if not pesos_iniciales:
            pesos = [random.uniform(-1, 1) for i in range(0, len(entrenamiento[0]))]
        else:
            pesos = pesos_iniciales
        
        epoch = 0

        while epoch < n_epochs:
            
            # o = umbral (w * x)
            # wi <- wi + n * xi (y - o)
            
            for j, _ in enumerate(entrenamiento):
                o = utils.umbral(utils.pesos_por_atributo(pesos, entrenamiento[j]))
                print(self.clases)
                print(o)
                pass

            epoch += 1
    
    def clasifica_prob(self, ej):
        pass
    
    def clasifica(self, ej):
        pass