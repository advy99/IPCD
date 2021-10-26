# -*- coding: utf-8 -*- 

"""
La traspuesta de una matriz se obtiene intercambiando filas y columnas. Escribe
una función que devuelva la traspuesta de una matriz.
"""

import generar_lista


def mostrar_matriz(matriz):
	for fila in matriz:
		for valor_columna in fila:
			print(valor_columna, end = " ")

		print()


def traspuesta(matriz):
	# con el * en matriz le decimos que cada elemento de la matriz va por separado
	# es decir, que cada fila es un elemento al que hacer el zip 
	# con esto si tenemos [[a, b], [c,d]] hará el zip con [a,b] y [c, d]
	# dando como resultado [a, c], [b, d]
	# para que zip nos devuelva el resultado en listas, usamos el * enfrente de zip
	resultado = [*zip(*matriz)]
	
	return resultado



if __name__ == "__main__":
	matriz = [ generar_lista.generar_lista(5) for i in range(5) ]

	print("Matriz original: ")
	mostrar_matriz(matriz)

	print("\nMatriz traspuesta: ")
	mostrar_matriz(traspuesta(matriz))


