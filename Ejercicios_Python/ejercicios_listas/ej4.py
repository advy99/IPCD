# -*- coding: utf-8 -*-

"""
Escribe una funcion combinar_listas(l1, l2) que devuelve una lista que 
este formada por todos los elementos de l1 y a continuacion todos los de 
l2.
"""

import generar_lista

def combinar_listas(l1, l2):
	# tambien podría haber sido 
	# return l1[:].extend(l2)

	resultado = []

	for elemento in l1:
		resultado.append(elemento)
	
	for elemento in l2:
		resultado.append(elemento)

	return resultado



if __name__ == "__main__":
	lista1 = generar_lista.generar_lista(5)
	lista2 = generar_lista.generar_lista(3)

	print("El resultado de combinar {} y {} es: {}".format(lista1, lista2, combinar_listas(lista1, lista2)))


