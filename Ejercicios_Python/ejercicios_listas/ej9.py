# -*- coding: utf-8 -*-

"""
Escribe una funcion suma_acumulada(numeros) a la que se le pase una lista
de numeros y devuelve una lista en la que el i-esimo se obtiene como la 
suma de los elementos de las posiciones 0 e i. 
"""

import generar_lista

def suma_acumulada(numeros):
	resultado = []

	# inicializamos por si está vacio
	if len(numeros) > 0:
		resultado.append(numeros[0])

	for i in range(1, len(numeros)):
		resultado.append(numeros[i] + resultado[-1])
	
	return resultado

if __name__ == "__main__":
	lista = generar_lista.generar_lista(4)

	print("La suma acumulada de {} es: {}".format(lista, suma_acumulada(lista)))




