# -*- coding: utf-8 -*-

"""
Escribe una función elimina_vocales(palabra) que elimine todas las vocales 
que aparecen en la palabra.
"""


VOCALES = "aeiouáéíóúAEUIOÁÉÍÓÚ"

def elimina_vocales(palabra):
	resultado = ""

	for caracter in palabra:
		if caracter not in VOCALES:
			resultado += caracter

	return resultado

if __name__ == "__main__":
	palabra = "prueba"
	print("{} sin vocales es {}".format(palabra, elimina_vocales(palabra)))


