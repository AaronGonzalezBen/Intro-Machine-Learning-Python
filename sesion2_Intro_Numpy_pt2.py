#-------------------------------------------INTRODUCCION A MANEJO DE DATOS CON NUMPY, PANDAS Y MATPLOTLIB---------------------------------------------------#

# Introducción a NumPy

import numpy as np

# Crear una matriz de unos - 3 filas 4 columnas
unos = np.ones((3,4))
print(unos)
print()

# Crear una matriz de ceros - 3 filas 4 columnas
ceros = np.zeros((3,4))
print(ceros)
print()

# Crear una matriz con valores aleatorios
aleatorios = np.random.random((2,2))
print(aleatorios)
print()

# Crear una matriz vacía
vacia = np.empty((3,2))
print(vacia)
print()

# Crear una martriz con un solo valor
full = np.full((2,2),8)
print(full)
print()

# Crear matrices con valores espaciados uniformemente
# arreglo de 0 a 30 con pasos de 5
espacio1 = np.arange(0,30,5)
print(espacio1)
print()

# arreglo de 0 a 2 partido en 5 elementos
espacio2 = np.linspace(0,2,5)
print(espacio2)
print()

# Crear matrices identidad
identidad1 = np.eye(4,4)
print(identidad1)
print()

identidad2 = np.identity(4)
print(identidad2)
print()

# Hallar dimensiones de matrices
a = np.array([(1,2,3),(4,5,6)])
print(a.ndim)
print()

# Conocer el tipo de datos
b = np.array([(1,2,3)])
print(b.dtype)
print()

# Encontrar el tamaño y forma de la matriz
c = np.array([(1,2,3,4,5,6)])
print(c.size)
print(c.shape)
print()

# Cambio de tamaño y forma de las matrices
# Convertir una matriz 2x3 a una 3x2
a = np.array([(8,9,10),(11,12,13)])
print(a)
print()

a = a.reshape(3,2)
print(a)
print()

# Extraer un solo valor de la matriz - el valor ubicado en la fila 0 columna 2
a = np.array([(1,2,3,4),(3,4,5,6)])
print(a[0,2])
print()

# Extraer los valores de todas las filas ubicados en la columna 3
a = np.array([(1,2,3,4),(3,4,5,6)])
print(a[0:,2])
print()

# Encontrar el mínimo, máximo y la suma
a = np.array([2,4,8])
print(a.min())
print(a.max())
print(a.sum())
print()

# Calcular la raiz cuadrada y la desviación estandar
a = np.array([(1,2,3),(3,4,5)])
print(np.sqrt(a))
print(np.std(a))
print()

# Calcular la suma, resta, multiplicacion y división de matrices
x = np.array([(1,2,3),(3,4,5)])
y = np.array([(1,2,3),(3,4,5)])
print(x+y)
print()
print(x-y)
print()
print(x*y)
print()
print(x/y)
print()

# Introducción a pandas
# Creacion de un dataframe
import pandas as pd

# Forma 1
data = np.array([['','Col1','Col2'],['Fila1',11,22],['Fila2',33,44]])
print(pd.DataFrame(data = data[1:,1:], index = data[1:,0], columns = data[0,1:]))
print()

# Forma 2
df = pd.DataFrame(np.array([[1,2,3],[4,5,6],[7,8,9]]))
print('DataFrame')
print(df)
print()

# Creación de una serie (diccionarios con indices)
series = pd.Series({"Argentina": "Buenos Aires", 
                    "Chile": "Santiago de Chile", 
                    "Colombia": "Bogotá", 
                    "Perú": "Lima"})
print('Series: ')
print(series)
print()

# Operaciones básicas para trabajar con DataFrames

# Forma del DataFrame
print('Forma del DataFrame: ')
print(df.shape)
print()

# Altura del DataFrame
print('Altura del DataFrame: ')
print(len(df.index))
print()

# Estadisiticas del DataFrame
print('Estadisticas del DataFrame: ')
print(df.describe())
print()

# Media de las columnas del DataFrame
print('Media de las columnas del DataFrame: ')
print(df.mean())
print()

# Correlación del DataFrame
print('Correlación del DataFrame: ')
print(df.corr())
print()

# Conteo de los datos del DataFrame
print('Conteo de los datos del DataFrame: ')
print(df.count())
print()

# Valor más alto de cada columna del DataFrame
print('Valor más alto de cada columna del DataFrame: ')
print(df.max())
print()

# Valor mínimo de cada columna del DataFrame
print('Valor mínimo de cada columna del DataFrame: ')
print(df.min())
print()

# Mediana de cada columna del DataFrame
print('Mediana de cada columna del DataFrame: ')
print(df.median())
print()

# Desviación estándar de cada columna del DataFrame
print('Desviación estándar de cada columna del DataFrame: ')
print(df.std())
print()

# Seleccionar la primera columna del DataFrame
print('Primera columna del DataFrame')
print(df[0])

# Seleccionar dos columnas del DataFrame
print('Dos columna del DataFrame')
print(df[[0,1]])

# Seleccionar el valor de la primera fila y última columna del DataFrame
print('Valor de la primera fila y última columna del DataFrame')
print(df.iloc[0][2])

# Seleccionar los valores de la primera fila del DataFrame a través del # de fila
print('Valores de la primera fila del DataFrame')
print(df.loc[0])

# Seleccionar los valores de la primera fila del DataFrame a través del indice
print('Valores de la primera fila del DataFrame')
print(df.iloc[0,:])

#df = pd.read_csv('train.csv')

# Verificar si hay datos nulos en el DataFrame
print('Datos nulos en el DataFrame')
print(df.isnull())
print()

# Suma de datos nulos en un DataFrame
print('Suma datos nulos en el DataFrame')
print(df.isnull().sum())
print()

# Eliminar filas donde haya valores perdidos
#pd.dropna()

# Eliminar las columnas donde hayan valores perdidos
#df.dropna(axis = 1)

# Reemplazar valores nulos
# df.fillna(x)

# Reemplazar valores perdidos con la media
print('Reemplazar los valores perdidos por la media: ')
print(df.fillna(df.mean()))
print()

# Introducción a Matplotlib
import matplotlib.pyplot as plt

a = [1,2,3,4]
b = [11,22,33,44]

plt.plot(a, b, color = 'blue', linewidth = 3, label = 'linea')
plt.legend()
plt.show()

# Otro diagrama de linea
a = [3,4,5,6]
b = [5,6,3,4]

plt.plot(a,b)
plt.show()

# Diagrama de linea mas elaborado

# Definimos los datos
x1 = [3,4,5,6]
y1 = [5,6,3,4]
x2 = [2,5,8]
y2 = [3,4,3]
# Configurar las caracteristicas del gráfico
plt.plot(x1, y1, label = 'Linea 1', linewidth = 4, color = 'blue')
plt.plot(x2, y2, label = 'Linea 2', linewidth = 4, color = 'red')
# Definir tituloy nombres de ejes
plt.title('Diagrama de Lineas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
# Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()

# Diagrama de barras
# Definimos los datos
x1 = [0.25,1.25,2.25,3.25, 4.25]
y1 = [10,55,80,32,40]
x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]
# Configurar las caracteristicas del gráfico
plt.bar(x1, y1, label = 'Datos 1', width = 0.5, color = 'lightblue')
plt.bar(x2, y2, label = 'Datos 2', width = 0.5, color = 'orange')
# Definir titulo y nombres de ejes
plt.title('Grafico de barras')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
# Mostrar leyenda, cuadricula y figura
plt.legend()
plt.grid()
plt.show()

# Histogramas
# Definir los datos
a = [22,55,62,45,21,22,34,42,42,4,2,102,
     95,85,55,110,120,70,65,55,111,115,80,75,65,54,
     44,43,42,48]
bins = np.arange(0,100,10)
# Configurar las caracteristicas del grafico
plt.hist(a,bins, histtype = 'bar', rwidth = 0.8
         , color = 'lightgreen')
# Definir titulo y nombre de ejes
plt.title('Histograma')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
# Mostrar leyenda, cuadricula y figura
plt.grid()
plt.show()

# Diagrama de barras
# Definimos los datos
x1 = [0.25,1.25,2.25,3.25, 4.25]
y1 = [10,55,80,32,40]
x2 = [0.75, 1.75, 2.75, 3.75, 4.75]
y2 = [42, 26, 10, 29, 66]
# Configurar las caracteristicas del gráfico
plt.scatter(x1, y1, label = 'Datos 1', color = 'red')
plt.scatter(x2, y2, label = 'Datos 2', color = 'purple')
# Definir titulo y nombres de ejes
plt.title('Grafico de dispersión')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')
# Mostrar leyenda, cuadricula y figura
plt.legend()
plt.show()

# Grafico circular
# Definir los datos
dormir = [7,8,6,11,7]
comer = [2,3,4,3,2]
trabajar = [7,8,7,2,2]
recreacion = [8,5,7,8,13]
divisiones = [7,2,2,13]
actividades = ['Dormir', 'Comer', 'Trabajar'
               , 'Recreación']
colores = ['red','purple','blue','orange']
# Configurar las caracteristicas del grafico
plt.pie(divisiones, labels = actividades, colors = colores,
        startangle = 90, shadow = True
        , explode = (0.1,0,0,0), autopct = '%1.1f%%')
# Definir titulo
plt.title('Grafico circular')
# Mostrar figura
plt.show()
































