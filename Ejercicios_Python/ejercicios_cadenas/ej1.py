# -*- coding: utf-8 -*-


"""
Escribe una función contar_letras(palabra, letra) que devuelva 
el número de veces que aparece una letra en una palabra.

"""

def contar_letras(palabra, letra):
	num_apariciones = 0

	# simplemente cuando aparezca la letra, sumamos uno en el contador
	for caracter in palabra:
		if caracter == letra:
			num_apariciones += 1

	return num_apariciones

if __name__ == "__main__":
	print("En la palabra 'palabra' aparecen {} veces la letra 'a' y {} veces la letra 'r'".format(contar_letras("palabra", 'a'), contar_letras("palabra", 'r')))


