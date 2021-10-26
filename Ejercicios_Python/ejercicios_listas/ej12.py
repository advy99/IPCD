# -*- coding: utf-8 -*-


"""
Escribe una funcion suma_primer_digito(numeros) que devuelva la suma de 
los primeros digitos de todos los numeros de la lista que se pasa como argumento.
"""

import generar_lista

def suma_primer_digito(numeros):

	resultado = 0

	for numero in numeros:
		# paso a string el numero, escojo el primer elemento, y lo paso a entero
		# ese es el primer digito, así que lo sumo al resultado
		resultado += int(str(numero)[0])

	return resultado


if __name__ == "__main__":
	lista = generar_lista.generar_lista(5, 5, 1000)

	print("Si en la lista {} sumamos el primer digito de cada elemento obtenemos: {}".format(lista, suma_primer_digito(lista)))


