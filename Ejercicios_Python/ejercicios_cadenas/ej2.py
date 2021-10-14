# -*- coding: utf-8 -*-

"""
Escribe una función eliminar_letras(palabra, letra) que devuelva
una versión de palabra que no contiene el carácter letra.
"""

def eliminar_letras(palabra, letra):
	resultado = ""	
	
	# para eliminar una letra, simplemente vamos copiando
	# caracter a caracter, saltando la letra que nos dan por parametro
	for caracter in palabra:
		if caracter != letra:
			resultado = resultado + caracter

	return resultado

