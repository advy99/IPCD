# -*- coding: utf-8 -*-

"""
Escribe una funcion dispersa(v) a la que se le pasa una lista representando un 
vector disperso y que devuelve el numero de elementos del vector junto con 
una lista de pares (pos, elem) con cada una de las posiciones en las que hay 
un elemento no nulo y el elemento
"""

import generar_lista

def dispersa(vector):
	
	resultado = []

	for i in range(len(vector)):
		if vector[i] != 0:
			resultado.append((i, vector[i]))
	

	return resultado, len(vector)

if __name__ == "__main__":
	lista = generar_lista.generar_lista(10)

	lista_indices = generar_lista.generar_lista(4, 0, 9)

	for elemento in lista_indices:
		lista[elemento] = 0

	print("Al vector {} le corresponde el vector disperso {}".format(lista, dispersa(lista)))




