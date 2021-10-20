# -*- coding: utf-8 -*-

"""
Escribe una función eco_palabra(palabra) que devuelva una cadena formada 
por palabra repetida tantas veces como sea su longitud. 
"""

def eco_palabra(palabra):
	return palabra * len(palabra) 


if __name__ == "__main__":
	print("El eco de hola es: ", eco_palabra("hola"))




