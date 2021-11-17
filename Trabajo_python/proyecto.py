# -*- coding: utf-8 -*-

#
# Bibliotecas que utilizaremos
#

import pandas as pd 
import sklearn as skl
import sklearn.model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#
# Funciones
#






def main():
	# leemos los datos
	datos = 	pd.read_csv("datos/SouthGermanCredit.csv", sep = " ")  
	print("Nombres de las columnas: ", datos.columns)
	predictores = datos.iloc[:,:-1]
	etiquetas = datos.kredit

	print("Echamos un ojo a los predictores: ")
	print(predictores.head())
	print("---")
	print("Miramos las etiquetas: ")
	print(etiquetas.head())

	train, test = skl.model_selection.train_test_split(predictores, etiquetas)




if __name__ == "__main__":
	main()


