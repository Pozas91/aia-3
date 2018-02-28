# -*- coding: utf-8 -*-

import time
import math
import pickle
import matplotlib.pyplot as plt

"""
Función utilizada para capturar el momento en el que comienza todo.
"""


def comienzo_tiempo_ejecucion():
    # Variable usada para medir los tiempos de ejecución
    start_time = time.time()
    return start_time


"""
Función utilizada para calcular los tiempos de ejecución de la aplicación en base al comienzo.
"""


def tiempo_ejecucion_obtenido(start_time):
    # Tiempo de ejecución obtenido
    print("Tiempo de ejecución en segundos: --- %s seconds ---" % (time.time() - start_time))


"""
La función umbral devuelve 1 si x es mayor o igual que 0, en caso contrario, devuelve 0.
"""


def umbral(x: float) -> int:
    return 1 if x >= 0 else 0


"""
Devuelve la suma del resultado de multiplicar cada valor del atributo con su peso correspondiente.
"""


def pesos_por_atributo(w: list, x: list) -> float:
    if len(w) != len(x):
        raise ValueError("El tamaño de w y de x debe ser el mismo.")

    return sum([a * b for a, b in zip(x, w)])


"""
Devuelve la función sigma en z.
"""


def sigma(z: float) -> float:
    return 1 / (1 + math.exp(z))


"""
Devuelve la derivida de sigma en z.
"""


def derivada_sigma(z: float) -> float:
    return sigma(z) * (1 - sigma(z))


"""
Devuelve la tasa de aprendizaje dado el número de epoch en que se encuentra.
"""


def rate_decay(tasa_inicial: float, epoch: int) -> float:
    return tasa_inicial + 2 / ((epoch ** 2) ** (1 / 3))


"""
Convirte republicano en 0 y demócrata en 1
"""
def convierte_republicano_democrata(voto):
    if voto == 'republicano':
        return 0
    else:
        return 1

"""
Vuelca la información de los pesos iniciales en el fichero indicado
"""


def guardar_pesos(nombre_del_fichero: str, data: list) -> None:
    with open(nombre_del_fichero, 'wb') as f:
        pickle.dump(data, f)


"""
Recupera la información de los pesos del fichero indicado
"""


def recuperar_pesos(nombre_del_fichero: str) -> list:
    try:

        with open(nombre_del_fichero, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


"""
Genera un gráfico dado un parámetro de entrada: errores
"""


def generar_grafico(errores):
    plt.plot(range(1, len(errores) + 1), errores, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de errores')
    plt.show()


"""
Función que dada una probabilidad entre 0 y 1 te devuelve el porcentaje de aproximación a la clase más cercana.
"""


def ponderar_probabilidad(x: float) -> float:
    if x > 0.5:
        return probabilidad_ascendiente(x)
    else:
        return probabilidad_descendiente(x)


"""
Función que dada una probabilidad entre 0 y 0.5, devuelve un porcentaje de proximidad
"""


def probabilidad_descendiente(x: float) -> float:
    return 100 - 200*x


"""
Función que dada una probabilidad entre 0.5 y 1, devuelve un porcentaje de proximidad
"""


def probabilidad_ascendiente(x: float) -> float:
    return 200*x - 100
