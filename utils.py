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
from matplotlib import pyplot as plt

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


def pesos_por_atributo(w: list, x: np.ndarray) -> float:
    if len(w) != len(x):
        raise ValueError("El tamaño de w y de x debe ser el mismo.")

    return sum([a * b for a, b in zip(x, w)])


"""
Devuelve la función sigma en z.
"""


def sigmoide(z: float) -> float:
    if z < -500:  # Para valores de z muy chicos la función sigmoide tiende a 1
        return 1
    elif z > 500:  # Para valores de z muy grandes lo función sigmoide tiende a 0
        return 0
    else:
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


def generar_grafico(errores, title, x_label='Epochs', y_label='Porcentaje de errores'):
    plt.plot(range(1, len(errores) + 1), errores, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
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
Genera un listado de pesos aleatorios [-1, 1]
"""


def generar_pesos_aleatorios(n_pesos: int) -> list:
    return [random.uniform(-1, 1) for _ in range(0, n_pesos)]


"""
Genera un conjunto completo de ejemplos aleatorios linealmente separables, devuelve como resultado la
tupla(ejemplos, clases)
"""


def generar_conjunto_separable(total_atributos: int, total_elementos: int, clases: np.ndarray) -> (list, list):
    total_ejemplos = []
    total_clases = []
    w = generar_pesos_aleatorios(total_atributos)

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


def generar_conjunto_no_separable(clases_separables: list, clases: np.ndarray, porcentaje: float):
    total_clases = len(clases_separables)
    total_cogidas = round(total_clases * porcentaje)
    clases_no_separables = deepcopy(clases_separables)

    indices = [i for i in range(0, total_clases)]
    random.shuffle(indices)

    indices_cogidos = [indices[i] for i in range(0, total_cogidas)]

    for i in indices_cogidos:
        indice_clase = clases.tolist().index(clases_no_separables[i])
        clases_no_separables[i] = clases[0] if indice_clase == 1 else clases[1]

    return clases_no_separables


"""
Devuelve la tupla (datos_normalizados, norma) una matriz de datos dada.
"""


def normalizar_si_es_necesario(datos, normalizar: bool):
    means = None
    std = None

    if normalizar:

        # Hacemos la medias por columnas
        means = np.mean(datos, axis=0)
        std = np.std(datos, axis=0)

        for i, _ in enumerate(datos):
            for j, _ in enumerate(datos[i]):
                datos[i][j] = np.subtract(datos[i][j], means[j])
                datos[i][j] = np.divide(datos[i][j], std[j])

    return datos, means, std


"""
Devuelve la tupla (datos_normalizados, norma) para una fila de datos dada.
"""


def normalizar_fila_si_es_necesario(fila, normalizar: bool, means, std):
    if normalizar:
        for i, _ in enumerate(fila):
            fila[i] = np.subtract(fila[i], means[i])
            fila[i] = np.divide(fila[i], std[i])

    return fila


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


"""
Genera una gráfica para representar en una nube de puntos los datos cargados del dataset
"""


def representacion_grafica(datos, caracteristicas,
                           objetivo, clases, c1, c2):
    for tipo, marca, color in zip(range(len(clases)), "soD", "rgb"):
        plt.scatter(datos[objetivo == tipo, c1],
                    datos[objetivo == tipo, c2],
                    marker=marca, c=color)
    plt.xlabel(caracteristicas[c1])
    plt.ylabel(caracteristicas[c2])
    plt.legend(clases)
    plt.show()
