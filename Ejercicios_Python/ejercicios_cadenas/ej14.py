# -*- coding: utf-8 -*-

"""
Escribe una función orden_alfabetico(palabra) que determine si las letras que
forman palabra aparecen en orden alfabético. 
"""

def orden_alfabetico(palabra):
	estan_ordenadas = True
	valor_numerico_letra = -1

	if len(palabra) > 0:
		valor_numerico_letra = ord(palabra[0])

	indice = 1

	while estan_ordenadas and indice < len(palabra):

		nuevo_valor_letra = ord(palabra[indice])
		estan_ordenadas = valor_numerico_letra < nuevo_valor_letra

		valor_numerico_letra = nuevo_valor_letra

		indice += 1


	return estan_ordenadas


def probar_orden_alfabetico(palabra):
	if orden_alfabetico(palabra):
		print("{} tiene sus caracteres ordenados de forma alfabética".format(palabra) )
	else:
		print("{} NO tiene sus caracteres ordenados de forma alfabética".format(palabra) )


if __name__ == "__main__":
	probar_orden_alfabetico("abejo")
	probar_orden_alfabetico("abeja")

		




