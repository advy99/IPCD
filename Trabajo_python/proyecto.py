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
import sklearn.metrics

### modelos de sklearn a utilizar
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.neural_network

### para reducción de dimensionalidad
import sklearn.decomposition
import sklearn.discriminant_analysis

#
# Funciones
#


def entrenar_modelo(modelo, predictores, etiquetas, predictores_test = None, etiquetas_test = None, num_folds = 10, normalizar = True, porcentaje_test = 0.2):
	"""
	Funcion para entrenar un modelo.
	Parametros:
		modelo: modelo a entrenar
		predictores: Predictores a los que ajustar el modelo
		etiquetas: Etiquetas asociadas a los predictores
		predictores_test: Predictores para el conjunto de test, por defecto None
		etiquetas_test: Etiquetas asociadas al conjunto de test, por defecto None
		num_folds: Numero de folds para aplicar en validación cruzada, por defecto 10
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
		predictores, predictores_test, etiquetas, etiquetas_test = skl.model_selection.train_test_split(predictores,
																										etiquetas,
																										test_size = porcentaje_test)

	if normalizar:
		# utilizamos un standard scaler para normalizar (media 0 y desviación 1)
		escalado = skl.preprocessing.StandardScaler()
		escalado.fit(predictores)
		predictores = escalado.transform(predictores)
		predictores_test = escalado.transform(predictores_test)


	# hacemos la validación cruzada
	resultado = skl.model_selection.cross_validate(modelo, predictores,
													etiquetas, cv = num_folds,
													scoring = "accuracy",
													return_estimator = True)

	# de todos los resultados obtenidos, buscamos el indice del mejor modelo
	# en test de cada fold
	mejor_modelo = np.argmax(resultado["test_score"])

	# calculamos la precisión del modelo en train y test, en test prediciendo
	# con el mejor modelo obtenido
	train_accuraccy_cv = np.mean(resultado["test_score"])
	test_accuraccy = np.mean(resultado["estimator"][mejor_modelo].predict(predictores_test) == etiquetas_test)


	return modelo, train_accuraccy_cv, test_accuraccy





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

	# probamos PCA eliminando solo una componente
	modelo_pca = skl.decomposition.PCA()
	# modelo_pca = skl.discriminant_analysis.LinearDiscriminantAnalysis()
	modelo_pca.fit(predictores)
	print("Porcentajes de varianza explicados por cada atributo resultante de PCA: ")
	print(modelo_pca.explained_variance_ratio_)

	modelo_pca = skl.decomposition.PCA(n_components = 1)
	modelo_pca.fit(predictores)

	predictores_pca = modelo_pca.transform(predictores)

	print("Tamaño de los predictores tras PCA con MLE: ", predictores_pca.shape)


	print("Pasamos a probar con distintos modelos y PCA.")

	modelos = [skl.linear_model.LogisticRegression(),
			   skl.tree.DecisionTreeClassifier(),
			   skl.ensemble.RandomForestClassifier(),
			   skl.svm.SVC(),
			   skl.neural_network.MLPClassifier()]


	for model in modelo:
		# TODO: GridSearch y RandomSearch


	# entrenamos cada modelo con sus mejores parámetros
	for model in modelos:
		modelo, train_accuraccy, test_accuraccy = entrenar_modelo(model, predictores_pca, etiquetas)

		print("Accuraccy en train con ", type(modelo).__name__, ": ", train_accuraccy)
		print("Accuraccy en test: ", type(modelo).__name__, ": ", test_accuraccy)
		print()





if __name__ == "__main__":
	main()
