# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:50:22 2020

@author: Sumadre

Regresion lineal con sklearn

"""

# Importamos la libreria para Regresion Lineal
from sklearn import linear_model

# Declaramos las variables de entrenamiento y prueba
# Dependientes (y) e Independientes (x)
x_entrenamiento = varInd_entrenamiento
y_entrenamiento = varDep_entrenamiento
x_prueba = varInd_prueba
y_prueba = varDep_prueba

# Declaramos el algoritmo de Regresion Lineal
algoritmo = linear_model.LinearRegression()
# Entrenamos con las variables de entrenamiento
algoritmo.fit(x_entrenamiento, y_entrenamiento)
# Predecimos con los datos de prueba
algoritmo.predict(x_prueba)

# Para conocer la pendiente
a = algoritmo.coef_
# Para conocer la intercección
b = algoritmo.intercept_

# Evaluamos la precisión del modelo
precision = algoritmo.score(x_prueba, y_prueba)