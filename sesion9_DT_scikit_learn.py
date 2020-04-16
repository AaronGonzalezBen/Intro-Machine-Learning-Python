# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:35:49 2020

Arboles de decisión en scikit learn para Regresión

Configuraciones del algoritmo DecisionTreeRegresor

- Criterion = Criterio de minimizacion de error, por defecto es 
MSE.
- splitter = división de cada uno de los nodos del arbol
- max_depth = profundidad maxima del arbol

@author: AARON
"""

from sklearn.tree import DecisionTreeRegresor

# 1. Separar datos de entrenamiento y prueba
x_entrenamiento = variablesIndependientes_entrenamiento
y_entrenamiento = variableDependiente_entrenamiento
x_prueba = variablesIndependientes_prueba
y_prueba = variableDependiente_prueba

# 2. Definir el algoritmo
algoritmo = DecisionTreeRegresor()
# 3. Entrenar el algoritmo
algoritmo.fit(x_entrenamiento, y_entrenamiento)
# 4. Predecir con el algoritmo
algoritmo.predict(x_prueba)

# 5. Evaluar la precisión del modelo
precision = algoritmo.score(x_prueba, y_prueba)