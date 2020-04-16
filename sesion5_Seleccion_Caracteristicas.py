# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 18:26:28 2020

@author: Alangobe

Métodos de selección de características

Necesario para conjunto de datos muy grandes

- Con el uso de todas las variables se genera mucho ruido
- Tarda más tiempo el entrenamiento

Se trata de seleccionar las variables más relevantes en
el conjunto de datos.

1. Métodos de filtro:
    - Conjunto con todas las caracteristicas (según la correlación con el target)
    - Selección del mejor subconjunto
    - Algoritmo de ML
    - Evaluación de rendimiento
    
    Caracteristicas/Predicción      Continuo           Categórico
    Continuo                   Correlación de Pearson      LDA
    Categórico                       Anova            Chi-cuadrado
    
    a. Correlación de Pearson: Medida para cuantificar la dependiencia lineal entre 2 variables continuas X y Y, varia entre 0 y 1.
    b. LDA: Analisis discriminante lineal, se usa para encontrar una combinación lineal de caracteristicas que caracteriza 2 o más
    clases, de una variable categórica.
    c. Anova: Análisis de la varianza, similar al LDA, pero oepra con una o más funciones dependientes categóricas y una independiente
    continua. Proporciona una medida estadística de si las medidas de varios grupos son iguales o no.
    d. Chi-cuadrado: Se aplica a los grupos de caracteristicas categóricas para evaluar la probabilidad de correlación o asociación
    entre ellos utilizando su distribución de frecuencia.
    
    Los métodos de filtros no eliminan la colinealidad.
    
2. Métodos de envoltura:
    - Conjunto con todas las caracteristicas -> Generar un subconjunto <-> Algoritmo de ML -> Evaluación de rendimiento
    
    a. Selección hacia adelante (Fordward pass): Método iterativo que comienza sin tener ninguna caracteristica del modelo. En cada iteración se agrega
    la función que mejor mejora el modelo hasta que la adición de una nueva variable no mejore el rendimiento del modelo.
    b. Eliminación hacia atrás (Backward selection): Se comienza con todas las características y se elimina la característica menos significativa
    en cada iteración, mejorando el rendimiento del modelo. Se repite esto hasta que no se observe ninguna mejora en la eliminación de características.
    c. Eliminación de características recursivas (Recursive Feature Elimination): Algoritmo de Optimización que busca encontrar el conjunto de funciones
    con mejor rendimiento. Crea repetidamente modelos y deja de lado la mejor o peor característica de rendimiento en cada iteración.
    
3. Métodos Integrados: Combinan las cualidades de los métodos anteriores, se implementa mediante algoritmos que tienen sus propios métodos
de selección de características incorporados.

    a. Regresión LASSO
    b. Regresión RIDGE
    
Diferencias:
    
1. Métodos de Filtros:
    - No incorporan un modelo de ML para determinar si una caracteristica es buena o mala.
    - Son mucho más rápidos ya que no implican la capacitación de los modelos.
    - Pueden no encontrar el mejor subconjunto de características cuando no hay suficientes
    datos para modelar la correlación estadística.
    - No conducirán a sobreajustes en la mayoría de los casos.
    
2. Métodos de Envoltura:
    - Incorporan un modelo de ML y lo capacitan para decidir si es esencial o no.
    - Son computacionalmente costosos, y en el caso de conjuntos de datos masivos no se recomiendan.
    - Siempre pueden proporcionar el mejor subconjunto de características debido a su naturaleza exhaustiva
    - En su modelo final de ML puede llevar a un sobreajuste

"""

