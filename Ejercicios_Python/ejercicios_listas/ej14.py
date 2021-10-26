# -*- coding: utf-8 -*-

"""
Escribe una funcion que saque de forma elatoria todas las cartas de una baraja
hasta que quede vacia. Para ello debe usar una lista que tenga inicialmente
todas las cartas.
"""

import random

def sacar_carta(baraja):
	
	while len(baraja) > 0:
		a_sacar = random.randint(0, len(baraja) - 1)
		print("Saco la carta {} de la baraja".format(baraja[a_sacar]))
		del baraja[a_sacar]


	


if __name__ == "__main__":
	# baraja con 40 cartas
	baraja = [x for x in range(1, 41)]
	sacar_carta(baraja)


