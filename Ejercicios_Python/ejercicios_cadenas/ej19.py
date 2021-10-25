# -*- coding: utf-8 -*-

"""
Escribe una funcion pangrama(frase) que determine si frase es o no un pangrama
"""

# utilizaremos el ejercicio 7 para pasar toda la frase a mayuscula
import ej7

def pangrama(frase):

	letras_aparecidas = dict()

	frase = ej7.mayusculas(frase)

	for letra in range(ord("A"), ord("Z") + 1):
		letras_aparecidas[chr(letra)] = False

	for letra in frase:
		letras_aparecidas[letra] = True

	return all(letras_aparecidas.values()) 
	

def probar_pangrama(frase):
	if pangrama(frase):
		print("\"{}\" es un pangrama".format(frase))
	else:
		print("\"{}\" NO es un pangrama".format(frase))


if __name__ == "__main__":
	probar_pangrama("Benjamin pidio una bebida de kiwi y fresa. Noe, sin verguenza, la mas exquisita champaña del menu.")
