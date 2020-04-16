# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:41:22 2020

@author: AARON
"""

#LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# PREPARAMOS LOS DATOS

# Importamos el dataset
boston = datasets.load_boston()

# ENTENDIMIENTO DE LA DATA

# Verificamos la información contenida en el dataset
print('Información del dataset: ')
print(boston.keys())
print()

# Verificamos las caracteristicas del dataset
print('Caracteristicas del dataset')
print(boston.DESCR)
print()

# Verificamos las dimensiones del dataset
print('Dimensión del dataset')
print(boston.data.shape)
print()

# Verificamos las dimensiones del dataset
print('Columnas del dataset')
print(boston.feature_names)
print()

# Separamos los datos de entrenamiento y prueba

# Seleccionamos la columna 6 del dataset
X_svr = boston.data[:, np.newaxis, 5]

# Defino los datos correspondientes al target
y_svr = boston.target

# Graficamos los datos
plt.scatter(X_svr,y_svr)
plt.show()

#### IMPLEMENTACION DE SVM REGRESION ####

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Separamos  los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size = 0.2)

# Defino el algoritmo SVM
svr = SVR(kernel = 'linear', C = 1.0, epsilon = 0.2)

# Entrenamos el modelo
svr.fit(X_train, y_train)

# realizamos la predicción
Y_pred = svr.predict(X_test)

# Graficamos los datos junto con la predicción
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color = 'red', linewidth = 3)
plt.show()

# Calculamos la precisión del modelo
print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()

print('Precisión del modelo: ')
print(svr.score(X_train, y_train))
















