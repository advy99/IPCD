# -*- coding: utf-8 -*-

"""
Escribe una función num_vocales(palabra) que devuelva el número de vocales 
que aparece en una palabra.
"""

VOCALES = "aeiouáéíóúAEUIOÁÉÍÓÚ"


def num_vocales(palabra):
	num_vocales_en_palabra = 0

	for letra in palabra:
		if letra in VOCALES:
			num_vocales_en_palabra += 1
	
	return num_vocales_en_palabra

if __name__ == "__main__":
	print("Hay {} vocales en la palabra: hola".format(num_vocales("hola")))
	print("Hay {} vocales en : hola mundo cruel".format(num_vocales("hola mundo cruel")))

