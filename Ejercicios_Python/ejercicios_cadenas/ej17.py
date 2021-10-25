# -*- coding: utf-8 -*-

"""
Escribe una funcion trocear(palabra, num) que devuelva una lista con trozos
de tamaño num de palabra.
"""


def trocear(palabra, num):
	trozos = []

	num_trozos = len(palabra) // num
	
	# si no puedo completar el ultimo trozo, añado uno
	# con el resto
	if len(palabra) % num != 0:
		num_trozos += 1

	trozos = [palabra[i * num: (i + 1) * num] for i in range(num_trozos)]

	return trozos


if __name__ == "__main__":
	print("Si troceo \"Hola mundo\" en trozos de longitud 4 obtengo: {}".format(trocear("Hola mundo", 4)))

