# -*- coding: utf-8 -*-

"""
Escribe una función mayusculas(palabra) que devuelva la palabra pasada 
a mayusculas
"""

DIFERENCIA_MAYUS_MINUS = ord('a') - ord('A') 

def es_minuscula(letra):
	return ord(letra) >= ord('a') and ord(letra) <= ord('z')

def convertir_a_mayuscula(letra):
	return chr(ord(letra) - DIFERENCIA_MAYUS_MINUS)

def mayusculas_minusculas(palabra):
	resultado = ""

	for caracter in palabra:
		# vamos a concatener el caracter por defecto
		# (si no es ni mayuscula ni minuscula)
		a_concatenar = caracter

		if es_minuscula(caracter):
			a_concatenar = convertir_a_mayuscula(a_concatenar)

		resultado += a_concatenar
	
	return resultado
			
if __name__ == "__main__":
	cadena =  "Hola Mundo!"
	print("Si en la cadena {} cambiamos todo por mayusculas nos queda: {}".format( 
			cadena, mayusculas_minusculas(cadena)))




