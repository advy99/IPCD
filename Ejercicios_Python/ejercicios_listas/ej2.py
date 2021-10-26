# -*- coding: utf-8 -*-

"""
Escribe una funcion contar_numeros_impares(numeros) que cuente la 
cantidad de numeros impares que hay en una lista
"""

import generar_lista

def contar_numeros_impares(numeros):
	resultado = 0

	for elemento in numeros:
		if elemento % 2 == 1:
			resultado += 1

	return resultado


if __name__ == "__main__":
	lista = generar_lista.generar_lista(5)

	print("En la lista {} hay {} elementos impares".format(lista, contar_numeros_impares(lista)))

