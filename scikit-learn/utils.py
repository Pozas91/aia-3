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