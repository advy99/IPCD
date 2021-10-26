# -*- coding: utf-8 -*-


"""
Escribe una funcion sum_nums_listas(numeros) que sume todos los numeros
de una lista. Compara el tiempo entre usar o no range. 
"""

import generar_lista
import time

def sum_nums_lista(numeros, usar_range = False):
	resultado = 0

	if usar_range:
		for i in range(len(numeros)):
			resultado += numeros[i]
	else:
		for elemento in numeros:
			resultado += elemento 

	return resultado

if __name__ == "__main__":

	lista_prueba = generar_lista.generar_lista(5)
	print("Resultado sobre la lista {}: {}".format(lista_prueba, sum_nums_lista(lista_prueba)))

	
	lista = generar_lista.generar_lista(10000000)

	inicio = time.time_ns()
	sum_nums_lista(lista)
	fin = time.time_ns()
	tiempo_sin_range = fin - inicio 

	inicio = time.time_ns()
	sum_nums_lista(lista, usar_range = True)
	fin = time.time_ns()
	tiempo_range = fin - inicio

	tiempo_range /= 1e9 
	tiempo_sin_range /= 1e9

	print("Con una lista de {} elementos:".format(len(lista)))
	print("\tUtilizando range se ha tardado {} segundos".format(tiempo_range))
	print("\tSin range se ha tardado {} segundos".format(tiempo_sin_range))




