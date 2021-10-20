# -*- coding: utf-8 -*-


"""
Escribe una función palindromo(frase) que determine si la frase es un 
palindromo. 
"""

def eliminar_letras(palabra, letra):
	resultado = ""	
	
	# para eliminar una letra, simplemente vamos copiando
	# caracter a caracter, saltando la letra que nos dan por parametro
	for caracter in palabra:
		if caracter != letra:
			resultado = resultado + caracter

	return resultado

def palindromo(frase):
	frase = eliminar_letras(frase, ' ')
	return frase == frase[::-1]


def probar_palindromo(palabra):
	if palindromo(palabra):
		print("{} es un palindromo".format(palabra))
	else:
		print("{} NO es un palindromo".format(palabra))

if __name__ == "__main__":
	probar_palindromo("hola")
	probar_palindromo("amor a roma")
	probar_palindromo("ana lleva al oso la avellana")


