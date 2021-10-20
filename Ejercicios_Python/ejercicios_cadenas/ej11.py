# -*- coding: utf-8 -*-

"""
Escribe una función comunes(palabra1, palabra2) que devuelva una cadena
formada por los caracteres comunes a las dos palabras.
"""

def comunes(palabra1, palabra2):
	resultado = ""

	for caracter in palabra1:
		if caracter in palabra2:
			resultado += caracter 

	return resultado

if __name__ == "__main__":
	print("Caracteres comunes a {} y {}: {}".format("Prueba", "Palabra", comunes("Prueba", "Palabra")))

