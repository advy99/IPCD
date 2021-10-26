# -*- coding: utf-8 -*-

"""
Escribe una funcion mezclar(la, lb) que dadas dos listas ordenadas devuelva 
una lista conteniendo los elementos de ambas listas ordenados de forma 
ascendente.
"""

import generar_lista
import ej4

def mezclar(la, lb):
	resultado = []

	indice_a = 0
	indice_b = 0

	while indice_a < len(la) and indice_b < len(lb):
		if la[indice_a] < lb[indice_b]:
			resultado.append(la[indice_a])
			indice_a += 1
		else:
			resultado.append(lb[indice_b])
			indice_b += 1


	if indice_a < len(la) - 1:
		resultado = ej4.combinar_listas(resultado, la[indice_a:])

	if indice_b < len(lb) - 1:
		resultado = ej4.combinar_listas(resultado, lb[indice_b:])

	return resultado


