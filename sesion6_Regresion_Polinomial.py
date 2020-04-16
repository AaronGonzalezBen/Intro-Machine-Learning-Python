# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:54:09 2020

@author: Alangobe

Regresión Polinomial

Extiende el modelo lineal al agregar predictores adicionales,
obtenidos al elevar cada uno de los predictores originales
a una potencia para generar un ajuste no lineal a los datos.
"""

# IMPORTAMOS LIBRERIAS
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt


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

###### PREPARAMOS LOS DATOS PARA LA REGRESION POLINOMIAL ###### 

# Seleccionamos solamente las columnas 5:8 del dataset
X_p = boston.data[:,np.newaxis, 5]
print(X_p)

# Defino los datos del target
y_p = boston.target

# Graficamos los datos correspondientes
plt.scatter(X_p, y_p)
plt.show()

# IMPLEMENTACIÓN DE REGRESIÓN POLINOMIAL
from sklearn.model_selection import train_test_split

# Separamos los datos de entrenamiento y prueba
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size = 0.2)

from sklearn.preprocessing import PolynomialFeatures

# Se define el grado del polinomio
# Se recomienda empezar con un grado bajo e ir subiendo
# hasta que se tenga el mejor rendimiento
poli_reg = PolynomialFeatures(degree = 2)

# Se transforman las características existentes en
# características de mayor grado
X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

# Se define el algoritmo a utilizar
pr = linear_model.LinearRegression()

# Entrenamos el modelo
pr.fit(X_train_poli, y_train_p)

# Realizo una predicción
Y_pred_pr = pr.predict(X_test_poli)

# Graficamos los datos junto con el modelo
plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_pr, color = 'red', linewidth = 3)
plt.show()

# Calculamos los coeficientes del modelo
print('DATOS DEL MODELO DE REGRESIÓN POLINOMIAL')
print()
print('Valor de la pendiente o coeficiente "a": ')
print(pr.coef_)
print()
print('Valor de la intersección o coeficiente "b": ')
print(pr.intercept_)
print()
print('La ecuación del modelo es igual a: ')
print('y = ', pr.coef_, 'x +', pr.intercept_)
print()

# Calculamos la precisión del modelo
# Devuelve el resultado de R^2
print('Precisión del modelo: ')
print(pr.score(X_train_poli, y_train_p))





























