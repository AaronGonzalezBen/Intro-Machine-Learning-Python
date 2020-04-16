# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 21:35:43 2020

Cómo generar métricas de desempeno de los modelos con 
scikit-learn

@author: AARON
"""

#### MATRIZ DE CONFUSIÓN

from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_test, y_pred)
print(matriz)

#### EXACTITUD

from sklearn.metrics import accuracy_score

exactitud = accuracy_score(y_test, y_pred)
print(exactitud)

##### PRECISION

from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(precision)

#### SENSIBILIDAD

from sklearn.metrics import recall_score

sensibilidad = recall_score(y_test, y_pred)
print(sensibilidad)

#### PUNTAJE F1

from sklearn.metrics import f1_score

puntaje = f1_score(y_test, y_pred)
print(puntaje)

