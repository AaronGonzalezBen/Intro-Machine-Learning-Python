# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:40:54 2020

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

# Preparamos los datos

# Seleccionamos la columna 6 deldataset
X_adr = boston.data[:, np.newaxis, 5]

# Definimos el target
y_adr = boston.target

#Graficamos los datos
plt.scatter(X_adr, y_adr)
plt.show()

#### Implementación del arbol de decisión #####

from sklearn.model_selection import train_test_split

# Separamos los datos de entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr, test_size = 0.2)

from sklearn.tree import DecisionTreeRegressor

# Defino el algoritmo a utilizar
adr = DecisionTreeRegressor(max_depth = 10)

# Entreno el modelo
adr.fit(X_train, y_train)

# Realizamos la predicción
Y_pred = adr.predict(X_test)

# Graficamos los datos de prueba junto con la predicción
X_grid = np.arange(min(X_test), max(X_test), 0.1) # declaramos un array de X
X_grid = X_grid.reshape((len(X_grid),1))    # lo transformamos a columna
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color = 'red', linewidth = 3)
plt.show()

# Calculamos la precisión del modelo
print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()

print('Precisión del modelo para datos de entrenamiento: ')
print(adr.score(X_train, y_train))
























