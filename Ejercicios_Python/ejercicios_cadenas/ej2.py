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

if __name__ == "__main__":
	print("Si a 'palabra' le quitamos la letra 'a' nos queda: {}".format(eliminar_letras("palabra", 'a')))


