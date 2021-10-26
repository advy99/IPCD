# -*- coding: utf-8 -*-

"""
Escribe una funcion cadena_mas_larga(cadenas) a la que se la pasa una lista
de palabras y que devuelva la palabra mas larga. 
"""

def cadena_mas_larga(cadenas):
	# usamos una funcion lambda, invertimos al ordenar porque si no sería 
	# la cadena más corta
	return sorted(cadenas, key = lambda x: len(x), reverse = True)[0]

if __name__ == "__main__":
	lista = ["Hola", "grandioso", "mundo", "cruel"]

	print("La cadena más larga de {} es {}".format(lista, cadena_mas_larga(lista)))





