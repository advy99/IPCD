# -*- coding: utf-8 -*-


"""
Escribe una función es_triple_doble(palabra) que determine si la palabra 
tiene tres pares de letras consecutivos.
"""

def es_triple_doble(palabra):
	# si no hay seis caracteres, no puede haber triple doble
	MAX_LONGITUD = 6
	suficiente_longitud = len(palabra) >= MAX_LONGITUD
	hay_triple_doble = False

	if suficiente_longitud:
		indice = 0

		while not hay_triple_doble and indice < len(palabra):
			inicio_comprobacion = 1
			hay_triple_en_caracter_indice = True 
			while hay_triple_en_caracter_indice and inicio_comprobacion < MAX_LONGITUD and inicio_comprobacion + indice < len(palabra):
				# compruebo que el segundo caracter de cada pareja es igual al caracter anterior
				hay_triple_en_caracter_indice = palabra[indice + inicio_comprobacion] == palabra[indice + inicio_comprobacion - 1]

				inicio_comprobacion += 2

			# si ha encontrado un triple doble comprobando todas las veces necesarias
			# la segunda condicion es por si se queda sin caracteres y sale antes de lo esperado
			# con una caso en el que si que tenga dos repetidas al final
			hay_triple_doble = hay_triple_en_caracter_indice and inicio_comprobacion >= MAX_LONGITUD
			indice += 1
				
			

	return hay_triple_doble
		
	
def probar_es_triple_doble(palabra):
	if es_triple_doble(palabra):
		print("Hay triple doble en {}".format(palabra))
	else:
		print("NO hay triple doble en {}".format(palabra))

if __name__ == "__main__":
	probar_es_triple_doble("abgghhkklme")
	probar_es_triple_doble("abgghkklme")
	probar_es_triple_doble("abgghhklme")







