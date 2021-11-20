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
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network

### para reducción de dimensionalidad
import sklearn.decomposition

#
# Funciones
#


def entrenar_modelo(modelo, predictores, etiquetas, predictores_test = None, etiquetas_test = None, normalizar = True, porcentaje_test = 0.2):
	"""
	Funcion para entrenar un modelo.
	Parametros:
		modelo: modelo a entrenar
		predictores: Predictores a los que ajustar el modelo
		etiquetas: Etiquetas asociadas a los predictores
		predictores_test: Predictores para el conjunto de test, por defecto None
		etiquetas_test: Etiquetas asociadas al conjunto de test, por defecto None
		normalizar: Booleano para marcar si es necesario normalizar o no los predictores, por defecto True
		porcentaje_test: Porcentaje de datos que conformarán el conjunto de test si no se ha pasado el conjunto de test

	Devuelve:
		modelo: Modelo ajustado a los predictores
		train_accuraccy: Precisión obtenida en el conjunto de train
		test_accuraccy: Precisión obtenida en el conjunto de test

	En caso de no indicar conjuntos de test, se crearan con el tamaño especificado
	en porcentaje_test.

	"""
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


	return modelo, train_accuraccy, test_accuraccy





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

	modelo_RL, train_accuraccy_RL, test_accuraccy_RL = entrenar_modelo(skl.linear_model.LogisticRegression(), predictores, etiquetas)

	print("Accuraccy en train con regresión logistica: ", train_accuraccy_RL)
	print("Accuraccy en test con regresión logistica: ", test_accuraccy_RL)
	print()


	modelo_DT, train_accuraccy_tree, test_accuraccy_tree = entrenar_modelo(skl.tree.DecisionTreeClassifier(), predictores, etiquetas)

	print("Accuraccy en train con un árbol de decisión: ", train_accuraccy_tree)
	print("Accuraccy en test con un árbol de decisión: ", test_accuraccy_tree)
	print()

	modelo_RF, train_accuraccy_RF, test_accuraccy_RF = entrenar_modelo(skl.ensemble.RandomForestClassifier(), predictores, etiquetas)

	print("Accuraccy en train con Random Forest: ", train_accuraccy_RF)
	print("Accuraccy en test con Random Forest: ", test_accuraccy_RF)
	print()

	modelo_SVC, train_accuraccy_svc, test_accuraccy_svc = entrenar_modelo(skl.svm.SVC(), predictores, etiquetas)

	print("Accuraccy en train con SVC: ", train_accuraccy_svc)
	print("Accuraccy en test con SVC: ", test_accuraccy_svc)
	print()

	modelo_MLP, train_accuraccy_MLP, test_accuraccy_MLP = entrenar_modelo(skl.neural_network.MLPClassifier(), predictores, etiquetas)

	print("Accuraccy en train con MLP: ", train_accuraccy_MLP)
	print("Accuraccy en test con MLP: ", test_accuraccy_MLP)
	print()




if __name__ == "__main__":
	main()