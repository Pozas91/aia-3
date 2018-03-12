# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from utils import *


# =============================================================================
# CARGAMOS EL DATASET
# =============================================================================
breast_cancer = load_breast_cancer()


# =============================================================================
# INICIALIZAMOS
# =============================================================================
X_cancer, y_cancer = breast_cancer.data, breast_cancer.target
X_names, y_names = breast_cancer.feature_names, breast_cancer.target_names

representacion_grafica(breast_cancer.data,X_names,y_cancer,y_names,0,1)


# =============================================================================
# CONJUNTO DE ENTRENAMIENTO
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_cancer,y_cancer,test_size = 0.15)

normalizador = StandardScaler().fit(X_train)

Xn_train = normalizador.transform(X_train)


# =============================================================================
# ALGORITMO KNN
# =============================================================================
knn = KNeighborsClassifier(n_neighbors=11)

# Método fit
knn.fit(Xn_train,y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
knn.predict(Xn_test)

# Método score
score_knn = knn.score(Xn_test,y_test)


# =============================================================================
# ALGORITMO LogisticRegression
# =============================================================================
lr = linear_model.LinearRegression()

# Método fit
lr.fit(Xn_train,y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
lr.predict(Xn_test)

# Método score
score_lr = lr.score(Xn_test,y_test)


# =============================================================================
# ALGORITMO ÁRBOL DE DECISIÓN
# =============================================================================
tree = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=5)

# Método fit
tree.fit(X_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
tree.predict(Xn_test)

# Método score
score_tree = tree.score(Xn_test,y_test)


# =============================================================================
# ALGORITMO RANDOM FOREST
# =============================================================================
random_forest = RandomForestClassifier(max_depth=2, random_state=0)

# Método fit
random_forest.fit(X_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
random_forest.predict(Xn_test)

# Método score
score_random_forest = random_forest.score(Xn_test,y_test)


# =============================================================================
# ALGORITMO LINEAR SVC
# =============================================================================
linear_svc = LinearSVC(random_state=0)

# Método fit
linear_svc.fit(X_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
linear_svc.predict(Xn_test)

# Método score
score_linear_svc = linear_svc.score(Xn_test,y_test)


# =============================================================================
# RESULTADOS OBTENIDOS
# =============================================================================
print("Porcentaje de aciertos sobre un conjunto de prueba (KNN): {0:.2f} %".format(score_knn * 100))
print("Porcentaje de aciertos sobre un conjunto de prueba (Linear Regression): {0:.2f} %".format(score_lr * 100))
print("Porcentaje de aciertos sobre un conjunto de prueba (Árbol de decisión): {0:.2f} %".format(score_tree * 100))
print("Porcentaje de aciertos sobre un conjunto de prueba (Random Forest): {0:.2f} %".format(score_random_forest * 100))
print("Porcentaje de aciertos sobre un conjunto de prueba (LinearSVC): {0:.2f} %".format(score_linear_svc * 100))
