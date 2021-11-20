# -*- coding: utf-8 -*-

#
# Bibliotecas que utilizaremos
#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

## modulos de sklearn
import sklearn as skl
import sklearn.model_selection
import sklearn.preprocessing

### modelos de sklearn a utilizar
import sklearn.linear_model
import sklearn.tree

#
# Funciones
#

def entrenar_modelo(modelo, predictores, etiquetas, predictores_test = None, etiquetas_test = None, normalizar = True, porcentaje_test = 0.2):

	if predictores_test == None or etiquetas_test == None:
		# separamos en entrenamiento y test dejando un 80% de los datos en entrenamiento
		# y un 20% en test
		predictores, predictores_test, etiquetas, etiquetas_test = skl.model_selection.train_test_split(predictores, etiquetas, test_size = porcentaje_test)

	if normalizar:
		escalado = skl.preprocessing.StandardScaler()
		escalado.fit(predictores)
		predictores = escalado.transform(predictores)
		predictores_test = escalado.transform(predictores_test)

	modelo.fit(predictores, etiquetas)

	train_accuraccy = np.mean(modelo.predict(predictores) == etiquetas)
	test_accuraccy = np.mean(modelo.predict(predictores_test) == etiquetas_test)


	return train_accuraccy, test_accuraccy





def main():
	# leemos los datos
	datos = 	pd.read_csv("datos/SouthGermanCredit.csv", sep = " ")
	print("Nombres de las columnas: ", datos.columns)
	predictores = datos.iloc[:,:-1].to_numpy()
	etiquetas = datos.kredit.to_numpy()

	print("Echamos un ojo a los predictores: ")
	print(predictores[0:5])
	print("---")
	print("Miramos las etiquetas: ")
	print(etiquetas[0:5])

	print("Tamaño de los predictores: ", predictores.shape)
	print("Tamaño de las etiquetas: ", etiquetas.shape)

	print("Pasamos a probar un modelo básico de regresión logistica.")
	modelo_RL = skl.linear_model.LogisticRegression()

	train_accuraccy, test_accuraccy = entrenar_modelo(modelo_RL, predictores, etiquetas)

	print("Accuraccy en train con regresión logistica: ", train_accuraccy)
	print("Accuraccy en test con regresión logistica: ", test_accuraccy)



if __name__ == "__main__":
	main()
