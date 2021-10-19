# -*- coding: utf-8 -*-

"""
Escribe una función mayusculas_minusculas(palabra) que devuelva una cadena
en la que las mayúsculas y las minúsculas estén al contrario
"""

DIFERENCIA_MAYUS_MINUS = ord('a') - ord('A') 


def es_mayuscula(letra):
	return ord(letra) >= ord('A') and ord(letra) <= ord('Z')


def es_minuscula(letra):
	return ord(letra) >= ord('a') and ord(letra) <= ord('z')

def convertir_a_minuscula(letra):
	# para converir a minuscula, a una mayuscula
	# le tengo que añadir la diferencia entre minusculas y mayusculas 
	# (las minusculas estan despeus)
	return chr(ord(letra) + DIFERENCIA_MAYUS_MINUS) 

def convertir_a_mayuscula(letra):
	return chr(ord(letra) - DIFERENCIA_MAYUS_MINUS)

def mayusculas_minusculas(palabra):
	resultado = ""

	for caracter in palabra:
		# vamos a concatener el caracter por defecto
		# (si no es ni mayuscula ni minuscula)
		a_concatenar = caracter

		if es_mayuscula(caracter):
			a_concatenar = convertir_a_minuscula(a_concatenar)
		elif es_minuscula(caracter):
			a_concatenar = convertir_a_mayuscula(a_concatenar)

		resultado += a_concatenar
	
	return resultado
			
if __name__ == "__main__":
	cadena =  "Hola Mundo"
	print("Si en la cadena {} cambiamos las mayusculas por minusculas y viceversa nos queda: {}".format( 
			cadena, mayusculas_minusculas(cadena)))




