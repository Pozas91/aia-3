# -*- coding: utf-8 -*-

import time
import math
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import glob
import img2pdf

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


def sigmoide(z: float) -> float:
    return 1 / (1 + math.exp(z))


"""
Devuelve la derivida de sigma en z.
"""


def derivada_sigma(z: float) -> float:
    return sigmoide(z) * (1 - sigmoide(z))


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


def generar_grafico(errores, title):
    plt.plot(range(1, len(errores) + 1), errores, marker='o')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Porcentaje de errores')
    plt.savefig('graficos/images/' + title + '.jpg')
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
    return 100 - 200 * x


"""
Función que dada una probabilidad entre 0.5 y 1, devuelve un porcentaje de proximidad
"""


def probabilidad_ascendiente(x: float) -> float:
    return 200 * x - 100


"""
Genera un ejemplo aleatorio de datos de tamaño n
"""


def generar_ejemplo_aleatorio(n: int) -> list:
    # Le quitamos uno, por que tenemos el término independiente
    return [1] + [random.randint(-1, 1) for _ in range(0, n - 1)]


"""
Genera un conjunto completo de ejemplos aleatorios linealmente separables, devuelve como resultado la
tupla(ejemplos, clases)
"""


def generar_conjunto_independiente(total_atributos: int, total_elementos: int, w: list, clases: list) -> (list, list):
    total_ejemplos = []
    total_clases = []

    for _ in range(0, total_elementos):
        # Cada ejemplo será de la misma longitud de un ejemplo que tenemos verificado
        x = generar_ejemplo_aleatorio(total_atributos)

        # Sacamos el resultado de esos pesos
        wx = pesos_por_atributo(w, x)

        # Sacamos la clase a la que pertenece para que sea linealmente separable
        c = clases[umbral(wx)]

        total_ejemplos.append(x)
        total_clases.append(c)

    return total_ejemplos, total_clases


"""
Genera un conjunto de clases inseparables dado un conjunto separable de clases
"""


def generar_conjunto_dependiente(clases_separables: list, clases: list, porcentaje: float):
    total_clases = len(clases_separables)
    total_cogidas = round(total_clases * porcentaje)
    clases_no_separables = deepcopy(clases_separables)

    indices = [i for i in range(0, total_clases)]
    random.shuffle(indices)

    indices_cogidos = [indices[i] for i in range(0, total_cogidas)]

    for i in indices_cogidos:
        indice_clase = clases.index(clases_no_separables[i])
        clases_no_separables[i] = clases[0] if indice_clase == 1 else clases[1]

    return clases_no_separables


"""
Devuelve la fila introducida normalizada
"""


def normalizar_fila(fila: list) -> (list, float, float):
    media = np.average(fila)
    std = np.std(fila)
    fila = np.subtract(fila, media)
    fila = np.divide(fila, std)
    return fila, media, std


"""
Devuelve la tupla (datos_normalizados, norma) una matriz de datos dada.
"""


def normalizar_si_es_necesario(datos, normalizar: bool):
    medias = None
    desviaciones_tipicas = None

    if normalizar:

        medias = [None for _ in range(len(datos))]
        desviaciones_tipicas = [None for _ in range(len(datos))]

        for i, _ in enumerate(datos):
            datos[i], medias[i], desviaciones_tipicas[i] = normalizar_fila(datos[i])

    return datos, medias, desviaciones_tipicas


"""
Devuelve la tupla (datos_normalizados, norma) para una fila de datos dada.
"""


def normalizar_fila_si_es_necesario(fila, normalizar: bool):
    # Se normaliza por la columna, no por la fila, por eso es necesario guardar la media y la std de la columna.
    media = None
    desviacion_tipica = None

    if normalizar:
        fila, media, desviacion_tipica = normalizar_fila(fila)

    return fila, media, desviacion_tipica


"""
Función utilizada para leer todas las imágenes en formato png y exportar a un único PDF
"""


def convierte_imagenes_PDF():
    filenames = [glob.glob("graficos/images/*.jpg")]

    with open("graficos/graficos.pdf", "wb") as f:
        for filename in filenames:
            f.write(img2pdf.convert(filename))


"""
Devuelve la lista de los índices aleatoria
"""


def random_indices(n: int) -> list:
    indices = [i for i in range(n)]
    random.shuffle(indices)
    return indices


"""
Genero una lista de ceros y uno en función de la clase seleccionada.
Esto se utiliza para enfrentar una clase con el resto.
"""


def genera_lista_one_vs_rest(clases, ejemplo_clases):
    aux_clases = []
    for clase in clases:
        if clase == ejemplo_clases:
            aux_clases.append('1')
        else:
            aux_clases.append('0')
    return aux_clases
