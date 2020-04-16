# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:05:17 2020

Para este caso se determinará a partir de una regresión logística 
si un conjunto de pacientes tienen cancer o no

@author: AARON
"""

# Importamos el dataset a utilizar
from sklearn import datasets

# Cargamos el dataset 
dataset = datasets.load_breast_cancer()
print(dataset)

# Exploración de datos
print("Información del dataset: ")
print(dataset.keys())
print()

# Verifico las caracteristicas del dataset
print("Caracteristicas del dataset: ")
print(dataset.DESCR)

# Seleccionamos los datos
X = dataset.data

# DEfino la target
y = dataset.target

#### Implementacion de regresión logistica ####
from sklearn.model_selection import train_test_split

# Separamos los datos de train y test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Se escalan todos los datos (normalizacion)

# Se importa la libreria de escalamiento
from sklearn.preprocessing import StandardScaler
# Se  llama el metodo
escalar = StandardScaler()
# Se aplica el método con los datos de entrenamiento
X_train = escalar.fit_transform(X_train)
# Se aplica el método con los datos de prueba
X_test = escalar.fit_transform(X_test)

# Se define el algoritmo a utilizar
from sklearn.linear_model import LogisticRegression

algoritmo = LogisticRegression()

# Entrenamiento
algoritmo.fit(X_train, y_train)

# Predicción
y_pred = algoritmo.predict(X_test)

#### Calculo de metricas ####

# Obtenemos la matriz de confusión
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión: ')
print(matriz)

# Calculo de la precisión del modelo
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print('Precisión del modelo: ')
print(precision)

# Calculo la exactitud del modelo
from sklearn.metrics import accuracy_score

exactitud = accuracy_score(y_test, y_pred)
print('Exactitud del modelo: ')
print(exactitud)

# Calculo la sensibilidad del modelo
from sklearn.metrics import recall_score

sensibilidad = recall_score(y_test, y_pred)
print('Sensibilidad del modelo: ')
print(sensibilidad)

# Puntaje F1 del modelo
from sklearn.metrics import f1_score

puntajef1 = f1_score(y_test, y_pred)
print('Puntaje F1 del modelo: ')
print(puntajef1)

# Curva ROC - AUC del modelo
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)
print('Curva ROC - AUC del modelo: ')
print(roc_auc)

# Resumen de métricas
print('Precisión del modelo: ', precision)
print('Exactitud del modelo: ', exactitud)
print('Sensibilidad del modelo: ', sensibilidad)
print('Puntaje F1 del modelo: ', puntajef1)
print('Curva  ROC - AUC del modelo: ', roc_auc)