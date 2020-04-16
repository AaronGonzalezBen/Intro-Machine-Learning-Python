# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:49:57 2020

@author: Aarón

Implementación de Regresión Lineal Múltiple

"""

# IMPORTAMOS LIBRERIAS
from sklearn import datasets, linear_model

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

# PREPARAMOS LOS DATOS PARA LA REGRESION LINEAL

# Seleccionamos solamente las columnas 5:8 del dataset
X_multiple = boston.data[:, 5:8]
print(X_multiple)

# Defino los datos del target
y_multiple = boston.target

# IMPLEMENTACION DE REGRESIÓN LINEAL MULTIPLE
from sklearn.model_selection import train_test_split

# Separamos los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size = 0.2)

# Definimos el algoritmo a utilizar
lr_multiple = linear_model.LinearRegression()

# Entrenamos el modelo
lr_multiple.fit(X_train, y_train)

# Realizamos una predicción
Y_pred_multiple = lr_multiple.predict(X_test)

# Calculamos los coeficientes del modelo
print('DATOS DEL MODELO DE REGRESIÓN LINEAL MULTIPLE')
print()
print('Valor de la pendiente o coeficiente "a": ')
print(lr_multiple.coef_)
print()
print('Valor de la intersección o coeficiente "b": ')
print(lr_multiple.intercept_)
print()
print('La ecuación del modelo es igual a: ')
print('y = ', lr_multiple.coef_, 'x +', lr_multiple.intercept_)
print()

# Calculamos la precisión del modelo
# Devuelve el resultado de R^2
print('Precisión del modelo: ')
print(lr_multiple.score(X_train, y_train))













































