# -*- coding: utf-8 -*-


"""
Escribe una funcion eliminar(l1, l2) que dadas dos listas devuelva una
lista en la que esten todos los elementos de l1 que no estan en l2
"""

import generar_lista

def eliminar(l1, l2):
	resultado = []

	for elemento in l1:
		if elemento not in l2:
			resultado.append(elemento) 

	return resultado


if __name__ == "__main__":
	l1 = generar_lista.generar_lista(5)
	l2 = generar_lista.generar_lista(5)

	print("Los elementos de {} que no están en {} son: {}".format(l1, l2, eliminar(l1, l2)))




