# -*- coding: utf-8 -*-

"""
Usando las funciones ord y chr construye una funcion a la que se le pase un 
string y devuelva su version encriptada con desplazamiento arbitrario,
encriptar(cad, desp)
"""

def encriptar(cad, desp):
	resultado = ""

	for letra in cad:
		letra_numero = ord(letra)

		if letra_numero >= ord('a') and letra_numero <= ord('z'):
			inicio_rango = 'a'
			fin_rango = 'z'
		else:
			inicio_rango = 'A'
			fin_rango = 'Z'

		caracter_final = letra

		# si es una letra le aplicamos la transformacion
		if (letra_numero >= ord('a') and letra_numero <= ord('z')) or (letra_numero >= ord('A') and letra_numero <= ord('Z')):
			# nos llevamos la letra al rango 0-27
			letra_final = ord(letra) - ord(inicio_rango)
			# le sumamos el desplazamiento, y le hacemos modulo el tamaño 
			# del alfabeto
			letra_final = (letra_final + desp) % (ord(fin_rango) - ord(inicio_rango) + 1)
			# nos llevamos la letra final al rango real donde están las letras
			caracter_final = chr( letra_final + ord(inicio_rango) )

		resultado += caracter_final


	return resultado

if __name__ == "__main__":
	cadena = "abcdefghijklmnopqrstuvwxyz"
	encriptada13 = encriptar(cadena, 13)
	encriptada4 = encriptar(cadena, 4)
	
	print("La cadena: \"{}\" encriptada con ROT-13 es: {}".format(cadena, encriptada13))
	print("La cadena: \"{}\" encriptada con ROT-4 es: {}".format(cadena, encriptada4))



