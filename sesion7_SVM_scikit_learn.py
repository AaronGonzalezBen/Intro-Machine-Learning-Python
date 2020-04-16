# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 20:32:20 2020

VECTORES DE SOPORTE - REGRESION

El objetivo es aprender a usar el modulo svm de Sickit learn

@author: AARON
"""

from sklearn.svm import SVR


# 1. Separar datos de entrenamiento y prueba
x_entrenamiento = variablesIndependientes_entrenamiento
y_entrenamiento = variableDependiente_entrenamiento
x_prueba = variablesIndependientes_prueba
y_prueba = variableDependiente_prueba

# 2. Definir el algoritmo
algoritmo = SVR()
# 3. Entrenar el algoritmo
algoritmo.fit(x_entrenamiento, y_entrenamiento)
# 4. Predecir con el algoritmo
algoritmo.predict(x_prueba)

"""

Configuración modulo SVM

- Configurar la constante C, esta es un parametro de la funcion SVM
por defecto es = 1

- Configurar Epsilon, tambien es un parametro de SVM

- Configurar el kernel, otro argumento de la funcion SVM

"""

# 5. Evaluar la precisión del modelo
precision = algoritmo.score(x_prueba, y_prueba)