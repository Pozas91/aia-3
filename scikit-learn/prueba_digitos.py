# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from datasets.digitdata import classes, training_data, training_classes, validation_data, validation_classes, \
    test_classes, test_data
from utils import *

import numpy as np

# =============================================================================
# CARGAMOS EL DATASET E INICIALIZAMOS
# =============================================================================

# Reducimos el cojunto de entrenamieto para poder dar unos resultados en un tiempo razonable.
# [:2000] -> Los 2000 primeros elementos
classes = np.array(classes)
y_names = classes
X_train = training_data[:2000]
y_train = [classes.tolist().index(clase) for clase in training_classes[:2000]]
X_test = test_data
y_test = [classes.tolist().index(clase) for clase in test_classes]

# =============================================================================
# CONJUNTO DE ENTRENAMIENTO
# =============================================================================
normalizador = StandardScaler().fit(X_train)

Xn_train = normalizador.transform(X_train)

# =============================================================================
# ALGORITMO KNN
# =============================================================================
knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [5, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

clf_knn = GridSearchCV(knn, param_grid)
# Método fit
clf_knn.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_knn.predict(Xn_test)

# Método score
score_knn = clf_knn.score(Xn_test, y_test)

print("Rendimiento (KNN): {0:.2f} %".format(score_knn * 100))
print("Mejores parámetros elegidos: {0}".format(clf_knn.best_params_))
print("*******************************************************")

# =============================================================================
# ALGORITMO LogisticRegression
# =============================================================================
lr = linear_model.LogisticRegression()

param_grid = {
    'penalty': ['l1', 'l2'],
    'dual': [True, False],
}

clf_lr = GridSearchCV(lr, param_grid)

# Método fit
clf_lr.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_lr.predict(Xn_test)

# Método score
score_lr = clf_lr.score(Xn_test, y_test)

print("Rendimiento (Logistic Regression): {0:.2f} %".format(score_lr * 100))
print("Mejores parámetros elegidos: {0}".format(clf_lr.best_params_))
print("*******************************************************")

# =============================================================================
# ALGORITMO ÁRBOL DE DECISIÓN
# =============================================================================
tree = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'random_state': [25, 50, 80, 100],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

clf_tree = GridSearchCV(tree, param_grid)

# Método fit
clf_tree.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_tree.predict(Xn_test)

# Método score
score_tree = clf_tree.score(Xn_test, y_test)

print("Rendimiento (Árbol de decisión): {0:.2f} %".format(score_tree * 100))
print("Mejores parámetros elegidos: {0}".format(clf_tree.best_params_))
print("*******************************************************")

# =============================================================================
# ALGORITMO RANDOM FOREST
# =============================================================================
random_forest = RandomForestClassifier()

param_grid = {
    'random_state': [25, 50, 80, 100],
    'max_depth': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'n_estimators': [8, 10, 12, 15, 30, 50, 100]
}

clf_random_forest = GridSearchCV(random_forest, param_grid)

# Método fit
clf_random_forest.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_random_forest.predict(Xn_test)

# Método score
score_random_forest = clf_random_forest.score(Xn_test, y_test)

print("Rendimiento (Random Forest): {0:.2f} %".format(score_random_forest * 100))
print("Mejores parámetros elegidos: {0}".format(clf_random_forest.best_params_))
print("*******************************************************")

# =============================================================================
# ALGORITMO LINEAR SVC
# =============================================================================
linear_svc = LinearSVC()

param_grid = {
    'random_state': [25, 50, 80, 100]
}

clf_linear_svc = GridSearchCV(linear_svc, param_grid)

# Método fit
clf_linear_svc.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_linear_svc.predict(Xn_test)

# Método score
score_linear_svc = clf_linear_svc.score(Xn_test, y_test)

print("Rendimiento (LinearSVC): {0:.2f} %".format(score_linear_svc * 100))
print("Mejores parámetros elegidos: {0}".format(clf_linear_svc.best_params_))
print("*******************************************************")

# =============================================================================
# ALGORITMO PERCEPTRÓN
# =============================================================================
perceptron = linear_model.SGDClassifier()

param_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
}

clf_perceptron = GridSearchCV(perceptron, param_grid)

# Método fit
clf_perceptron.fit(Xn_train, y_train)

# Método predict
Xn_test = normalizador.transform(X_test)
clf_perceptron.predict(Xn_test)

# Método score
score_perceptron = clf_perceptron.score(Xn_test, y_test)

print("Rendimiento (SGDClassifier): {0:.2f} %".format(score_perceptron * 100))
print("Mejores parámetros elegidos: {0}".format(clf_perceptron.best_params_))
