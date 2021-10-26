# -*- coding: utf-8 -*-

"""
Escribe una funcion contar_letras(palabra) que tome una palabra como argumento
y devuelva una lista de pares en la que aparece cada letra junto con el numero
de veces que aparece esa letra en la palabra.
"""


def contar_letras(palabra):

	contador = dict()

	for letra in palabra:
		if letra in contador:
			contador[letra] += 1
		else:
			contador[letra] = 1

	return list(contador.items())	


if __name__ == "__main__":
	print("La palabra {} contiene las siguientes letras: {}".format("patata", contar_letras("patata")))

