# -*- coding: utf-8 -*-

"""
Escribe una funcion anagrama(palabra1, palabra2) que determine si 
es un anagrama.
"""

def contar_letras(palabra):
	conteo = dict()

	for letra in palabra:
		if letra in conteo:
			conteo[letra] += 1
		else:
			conteo[letra] = 1

	return conteo

# es un anagrama si al contar las letras tenemos las mismas letras para una
# palabra y para la otra
def anagrama(palabra1, palabra2):
	return contar_letras(palabra1) == contar_letras(palabra2)


def probar_anagrama(palabra1, palabra2):
	if anagrama(palabra1, palabra2):
		print("{} y {} son un anagrama".format(palabra1, palabra2))
	else:
		print("{} y {} NO son un anagrama".format(palabra1, palabra2))
	
if __name__ == "__main__":
	probar_anagrama("marta", "trama")
	probar_anagrama("marta", "tramo")







