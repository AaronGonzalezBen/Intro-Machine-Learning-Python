# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:50:22 2020

@author: Sumadre

Regresión lineal para predicción de precios en Vivienda
Con el Boston Dataset

"""

# LIBRERÍAS
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# PREPARAMOS LOS DATOS

# Importamos el dataset
boston = datasets.load_boston()
#print(boston)
#print()

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

# PREPARAMOS LOS DATOS PARA LA REGRESION LINEAL

# Seleccionamos solamente la columna 5 del dataset
X = boston.data[:,np.newaxis, 5]

# Defino los datos del target
y = boston.target

# Graficamos los datos
plt.scatter(X,y)
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()

# IMPLEMENTACION DE REGRESIÓN LINEAL SIMPLE
from sklearn.model_selection import train_test_split

# Separamos los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Definimos el algoritmo a utilizar
lr = linear_model.LinearRegression()

# Entrenamos el modelo
lr.fit(X_train, y_train)

# Realizamos la predicción
Y_pred = lr.predict(X_test)

# Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color = 'red', linewidth = 3)
plt.title('Regresión Lineal Simple')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()
print()

# Calculamos los coeficientes del modelo
print('DATOS DEL MODELO DE REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a": ')
print(lr.coef_)
print('Valor de la intersección o coeficiente "b": ')
print(lr.intercept_)
print()
print('La ecuación del modelo es igual a: ')
print('y = ', lr.coef_, 'x +', lr.intercept_)
print()

# Calculamos la precisión del modelo
# Devuelve el resultado de R^2
print('Precisión del modelo: ')
print(lr.score(X_train, y_train))












































