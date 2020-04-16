# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:34:00 2020

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

#### PREPARAMOS LOS DATOS DE ENTRENAMIENTO Y PRUEBA

# Seleccionamos unicamente la columna 6 del dataset
X_bar = boston.data[:, np.newaxis,5]

# Defino el target
y_bar = boston.target

# Graficamos los datos
plt.scatter(X_bar, y_bar)
plt.show()

#### IMPLEMENTACION DE PARTICION EN DATOS

from sklearn.model_selection import train_test_split

# Separo los datos de train y test
X_train, X_test, y_train, y_test = train_test_split(X_bar, y_bar, test_size = 0.2)

# Importamos el algoritmo de Random Forest
from sklearn.ensemble import RandomForestRegressor

bar = RandomForestRegressor(n_estimators = 300,max_depth = 8)

#### APLICAMOS EL ALGORITMO A LOS DATOS DE ENTRENAMIENTO
bar.fit(X_train, y_train)


#### PREDECIMOS CON EL ALGORITMO RF
Y_pred = bar.predict(X_test)

# Graficamos los datos
X_grid = np.arange(min(X_test), max(X_test), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, bar.predict(X_grid), color = 'red', linewidth = 3)
plt.show()

# Calculo la precision
precision = bar.score(X_train, y_train)
print(precision)










