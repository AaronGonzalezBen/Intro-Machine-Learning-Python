#-------------------------------------------INTRODUCCION A MANEJO DE DATOS CON NUMPY---------------------------------------------------#

import numpy as np

a = np.array([1,2,3])
print("1D array: ")
print(a)
print()

b = np.array([(1,2,3),(4,5,6)])
print("2D array: ")
print(b)
print()

# Verificacion de memoria de numpy

import sys

S = range(1000)
print("Resultado lista de Python: ")
print(sys.getsizeof(S)*len(S))
print()

D = np.arange(1000)
print("Resultado NumPy array")
print(D.size*D.itemsize)
print()

# Comparacion numpy vs librerias tradicionales

import time

SIZE = 1000000

L1 = range(SIZE)
L2 = range(SIZE)
A1 = np.arange(SIZE)
A2 = np.arange(SIZE)

start = time.time()

# List comprehension - genera nuevas listas a partir de un for
# aplicado a otra lista
# nueva_lista = [expresion(i) for i in vieja_lista if filtro(i)]
# zip genera tuplas de los argumentos ingresados, elemento por elemento
# y los almacena en un objeto iterador
result = [(x,y) for x,y in zip(L1,L2)]
print("Resultado lista de Python")
print((time.time() - start)*1000)
print()

start = time.time()
result = A1 + A2
print("Resultado NumPy array: ")
print((time.time() - start)*1000)
