# -*- coding: utf-8 -*-

"""
Escribe una función es_inversa(palabra1, palabra2) que determine si 
una palabra es la misma que la otra pero con los caracteres en orden inverso. 
"""


def es_inversa(palabra1, palabra2):
	return palabra1 == palabra2[::-1]

if __name__ == "__main__":
	palabra1 = "absd"
	palabra2 = "dsba"

	if es_inversa(palabra1, palabra2):
		print("{} se escribe como {} pero con los caracteres en orden inverso".format(palabra1, palabra2))
	else:
		print("{} NO se escribe como {} pero con los caracteres en orden inverso".format(palabra1, palabra2))




