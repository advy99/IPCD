# -*- coding: utf-8 -*-

"""
Escribe una función inicio_fin_vocal(palabra) que determine si una palabra 
empieza y acaba con una vocal.
"""

VOCALES = "aeiouáéíóúAEUIOÁÉÍÓÚ"

def inicio_fin_vocal(palabra):

	resultado = None

	if len(palabra.split(' ')) != 1:
		print("AVISO: Se ha recibido una frase, no una única palabra.")
	else:
		resultado = palabra[0] in VOCALES and palabra[-1] in VOCALES

	return resultado


def comprobar_palabra(palabra):
	if inicio_fin_vocal(palabra):
		print("{} comienza y acaba por vocal.".format(palabra))
	else:
		print("{} NO comienza y acaba por vocal.".format(palabra))


if __name__ == "__main__":
	comprobar_palabra("hola")
	comprobar_palabra("ola")

		


