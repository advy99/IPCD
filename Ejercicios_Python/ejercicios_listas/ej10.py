# -*- coding: utf-8 -*-


"""
Escribe una función parejas(lista) que calcula las parejas distintas
de valores que aparecen en una lista
"""

import generar_lista

def parejas(lista):
	combinaciones = [ (x, y) for y in lista for x in lista ]

	resultado = []

	# nos quedamos con las combinaciones unicas
	for combinacion in combinaciones:
		if combinacion not in resultado:
			resultado.append(combinacion)

	return resultado


if __name__ == "__main__":
	lista = generar_lista.generar_lista(3)
	print(lista)
	print(parejas(lista))



