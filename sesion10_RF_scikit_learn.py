# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:26:54 2020

Bosques aleatorios - Regresión

Dentro del algoritmo de Bosques aleatorios tenemos

- n_estimators: # de arboles delbosque - por defecto 10
- criterion: funcion de error, en este caso se usa MAE
- max_depht: profundidad del arbol, si no se selecciona
se toma el mayor numero de ramificaciones del arbol
- max_features = # maximo de caracteristicas

@author: AARON
"""

from sklearn.ensemble import RandomForestRegressor

x_entrenamiento = variablesIndependientes_entrenamiento
y_entrenamiento = variableDependiente_entrenamiento
x_prueba = variablesIndependientes_prueba
y_prueba = variableDependiente_prueba

algoritmo = RandomForestRegressor
algoritmo.fit(x_entrenamiento, y_entrenamiento)
algoritmo.predict(x_prueba)

precision = algoritmo.score(x_prueba, y_prueba)
