# -*- coding: utf-8 -*-

"""
Escribe una función vocales(palabra) que devuelva las vocales 
que aparecen en una palabra.
"""

VOCALES = "aeiouáéíóúAEUIOÁÉÍÓÚ"


def vocales(palabra):
	vocales_en_palabra = set()

	for letra in palabra:
		if letra in VOCALES:
			vocales_en_palabra.add(letra)
	
	return vocales_en_palabra

if __name__ == "__main__":
	print("Las vocales que aparecen en hola son: {}".format(vocales("hola")))
	print("Las vocales que aparecen en hola mundo cruel son: {}".format(vocales("hola mundo cruel")))

