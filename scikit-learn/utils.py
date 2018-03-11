# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

"""
Genera una gráfica para representar en una nube de puntos los datos cargados del dataset
"""
def representacion_grafica(datos,caracteristicas,
    objetivo,clases,c1,c2):
    for tipo,marca,color in zip(range(len(clases)),"soD","rgb"):
        plt.scatter(datos[objetivo == tipo,c1],
                    datos[objetivo == tipo,c2],
                    marker=marca,c=color)
    plt.xlabel(caracteristicas[c1])
    plt.ylabel(caracteristicas[c2])
    plt.legend(clases)
    plt.show()
    

"""
Obtiene el porcentaje resultante de aplicar un algoritmo en un conjunto de datos
"""
def obtener_porcentaje(algoritmo, X_train, y_train, X_test, y_test, normalizador):
    # Método fit
    algoritmo.fit(X_train, y_train)
    
    # Método predict
    Xn_test = normalizador.transform(X_test)
    algoritmo.predict(Xn_test)
    
    # Método score
    score = algoritmo.score(Xn_test,y_test)
    
    return score