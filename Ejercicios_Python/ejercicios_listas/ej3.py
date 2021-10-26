# -*- coding: utf-8 -*-

"""
Escribe una funcion numeros_pares(numeros) que devuelva los numeros pares 
que hay en una lista. 
"""

import generar_lista

def numeros_pares(numeros):
	return [x for x in numeros if x % 2 == 0]


if __name__ == "__main__":
	lista = generar_lista.generar_lista(5)

	print("Los numeros pares de {} son: {}".format(lista, numeros_pares(lista)))





