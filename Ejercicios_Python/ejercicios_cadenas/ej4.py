# -*- coding: utf-8 -*-

"""
Escribe una función buscar(palabra, sub) que devuelva la posición 
en la que se puede encontrar sub dentro de la palabra, o -1 en caso de 
que no esté.
"""


def buscar(palabra, sub):
	indice = 0
	aparicion = 0
	encontrado = False

	LONGITUD_PALABRA = len(palabra)
	LONGITUD_SUBPALABRA = len(sub)

	while(indice < LONGITUD_PALABRA and not encontrado):
		if (palabra[indice] == sub[aparicion]):
			aparicion += 1
		else:
			indice -= aparicion
			aparicion = 0

		encontrado = aparicion == LONGITUD_SUBPALABRA

		indice += 1

	resultado = -1

	if encontrado:
		resultado = indice - aparicion

	return resultado


if __name__ == "__main__":
	print("Podemos encontrar \"undo\" en la posición {} de la cadena: Hola mundo ".format(buscar("Hola mundo", "undo")) )
	print("Podemos encontrar \"asd\" en la posición {} de la cadena: Hola mundo ".format(buscar("Hola mundo", "asd")) )
	print("Podemos encontrar \"mund \" en la posición {} de la cadena: Hola mundo ".format(buscar("Hola mundo", "mund ")) )
	print("Podemos encontrar \"Hola mundo\" en la posición {} de la cadena: Hola mundo ".format(buscar("Hola mundo", "Hola mundo")) )
	print("Podemos encontrar \"Hol\" en la posición {} de la cadena: Hola mundo ".format(buscar("Hola mundo", "Hol")) )



