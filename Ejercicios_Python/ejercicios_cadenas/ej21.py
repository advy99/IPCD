# -*- coding: utf-8 -*-

"""
Escribe una funcion suma_digitos(cad) que haga la suma de los digitos de un 
numero que está en cad. Modificar ahora la funcion para que también funcione
si cad es un int
"""


def suma_digitos(cad):
	numero = int(cad)

	resultado = 0

	while (numero > 0):
		resultado += numero % 10
		numero = numero // 10

	return resultado

def probar_suma_digitos(cad):
	print("La suma de los dígitos de {} es: {}".format(cad, suma_digitos(cad)))

if __name__ == "__main__":
	probar_suma_digitos("123")
	probar_suma_digitos(123)



