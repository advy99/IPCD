# -*- coding: utf-8 -*-


"""
Escribe una función todas_las_letras(palabra, letras) que determine si se 
han usado todos los caracteres de letras en palabra.
"""

def todas_las_letras(palabra, letras):
	contiene_todas_las_letras = True

	indice_letras = 0

	while contiene_todas_las_letras and indice_letras < len(letras):
		contiene_todas_las_letras = letras[indice_letras] in palabra
		indice_letras += 1

	return contiene_todas_las_letras


def probar_todas_letras(palabra, letras):
	if todas_las_letras(palabra, letras):
		print("La palabra \"{}\" contiene las siguientes letras: {}".format(palabra, letras))
	else: 
		print("La palabra \"{}\" NO contiene las siguientes letras: {}".format(palabra, letras))


if __name__ == "__main__":
	probar_todas_letras("hola", "ol")
	probar_todas_letras("hola mundo cruel", "abzspe")

