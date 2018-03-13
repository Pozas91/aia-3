# AIA 3 - Clasificadores Lineales

## Ficheros de pruebas

### main.py
Este fichero contiene la ejecución de los 5 tipos de clasificadores diferentes (sin contar con el One vs Rest),
se ejecuta sobre el conjunto de votos y da una comparación sobre el rendimiento y la clasificación de los 
distintos clasificadores lineales.

### prueba_pu.py
Este fichero contiene la ejecución del clasificador __perceptrón umbral__, ejecutado sobre el conjunto de votos.

### prueba.recb.py
Este fichero contiene la ejecución del clasificador __regresión error cuadrático batch__, ejecutado sobre el conjunto de votos.

### prueba.rece.py
Este fichero contiene la ejecución del clasificador __regresión error cuadrático estocástico__, ejecutado sobre el conjunto de votos.

### prueba.rvb.py
Este fichero contiene la ejecución del clasificador __regresión de verosimilitud batch__, ejecutado sobre el conjunto de votos.

### prueba.rve.py
Este fichero contiene la ejecución del clasificador __regresión de verosimilitud estocástico__, ejecutado sobre el conjunto de votos.

### prueba_ovr.py
Este fichero contiene la ejecución del clasificador __one vs. rest__, ejecutado sobre el conjunto de dígitos.

### manual_grid_search.py
Este fichero contiene la lógica para dado un clasificador, probar con diferente parámetros evaluando el rendimiento
y dando como resultado el mejor conjunto de parámetros encontrados para un rendimiento mayor. A diferencia de como
está implementando en __sklearn__, esta versión está hecha a mano.

### scikit-learn/prueba_cancer.py
Este fichero contiene los diferentes clasificadores de __scikit-learn__, ejecutados con __GridSearchCV__ sobre el conjunto de cáncer de mama.

### scikit-learn/prueba_digitos.py
Este fichero contiene los diferentes clasificadores de __scikit-learn__, ejecutados con __GridSearchCV__ sobre el conjunto de dígitos.

