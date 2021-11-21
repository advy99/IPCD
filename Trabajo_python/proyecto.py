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


def entrenar_modelo(modelo, predictores, etiquetas, predictores_test = None, etiquetas_test = None, num_folds = 10, porcentaje_test = 0.2):
	"""
	Funcion para entrenar un modelo.
	Parametros:
		modelo: modelo a entrenar
		predictores: Predictores a los que ajustar el modelo
		etiquetas: Etiquetas asociadas a los predictores
		predictores_test: Predictores para el conjunto de test, por defecto None
		etiquetas_test: Etiquetas asociadas al conjunto de test, por defecto None
		num_folds: Numero de folds para aplicar en validación cruzada, por defecto 10
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

	print("\nTamaño de los predictores: ", predictores.shape)
	print("Tamaño de las etiquetas: ", etiquetas.shape)

	# escalamos los datos antes de aplicar el PCA, ya que PCA calculará unos nuevos
	# predictores a partir de los actuales, y si no están escalados le dará más
	# importancia a unos que a otros
	# utilizamos un standard scaler para normalizar (media 0 y desviación 1)
	escalado = skl.preprocessing.StandardScaler()
	escalado.fit(predictores)
	predictores_escalados = escalado.transform(predictores)


	# aplicamos PCA, dejando tantas características como sean necesarias
	# para explicar un 90% de los datos
	modelo_pca = skl.decomposition.PCA(n_components = 0.9)
	modelo_pca.fit(predictores_escalados)
	# como vemos con una nos basta, hemos pasado de 20 predictores a 1
	print("Porcentajes de varianza explicados por cada atributo resultante de PCA final: ")
	print(modelo_pca.explained_variance_ratio_)

	predictores_pca = modelo_pca.transform(predictores_escalados)

	print("Tamaño de los predictores tras PCA: ", predictores_pca.shape)


	print("\nPasamos a buscar los mejores parámetros para cada modelo.")

	# todos los modelos que lo permiten trabajarán con NUM_CPUS para aplicar paralelismo y
	# realizar el proceso de busqueda de parámetros más rápido
	NUM_CPUS = 4
	modelos = [skl.linear_model.LogisticRegression(),
			   skl.tree.DecisionTreeClassifier(),
			   skl.ensemble.RandomForestClassifier(),
			   skl.svm.SVC(),
			   skl.neural_network.MLPClassifier()]

	# parametros para los modelos
	parametros = dict()

	parametros["LogisticRegression"] = {"C" : [0.001, 0.01, 0.1, 1, 10, 100],
										"solver": ["lbfgs", "sag", "liblinear"]}

	# un árbol de decisión es muy básico y no tiene muchos parámetros a ajustar
	parametros["DecisionTreeClassifier"] = {"criterion": ["gini", "entropy"]}

	parametros["RandomForestClassifier"] = {"n_estimators": [50, 100, 150, 200, 300, 500, 700],
											"criterion": ["gini", "entropy"],
											"bootstrap": [True, False]}

	parametros["SVC"] = {"C" : [0.001, 0.01, 0.1, 1, 10, 100],
						 "kernel": ["linear", "poly", "rbf", "sigmoid"],
						 "degree": [2, 3, 4, 5],
						 "gamma": ["scale", "auto", 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}


	parametros["MLPClassifier"] = {"activation": ["relu", "tanh", "logistic", "identity"],
								   "solver": ["adam", "sgd", "lbfgs"],
								   "alpha": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
								   "batch_size": ["auto", 10, 50, 100, 200, 500],
								   "learning_rate": ["constant", "invscaling", "adaptative"],
								   "max_iter" : [100, 200, 300, 400, 500]}

	mejores_estimadores_grid_search = dict()

	for modelo in modelos:
		# TODO: GridSearch y RandomSearch
		nombre_modelo = type(modelo).__name__
		grid_search = skl.model_selection.GridSearchCV(modelo, parametros[nombre_modelo])
		grid_search.fit(predictores_pca, etiquetas)
		print("El mejor estimador encontrado para el modelo ", nombre_modelo, " usando GridSearchCV es: ")
		print(grid_search.best_estimator_)
		mejores_estimadores_grid_search[nombre_modelo] = grid_search.best_estimator_


	mejores_estimadores_randomized_search = dict()

	for modelo in modelos:
		# TODO: GridSearch y RandomSearch
		nombre_modelo = type(modelo).__name__
		grid_search = skl.model_selection.GridSearchCV(modelo, parametros[nombre_modelo])
		grid_search.fit(predictores_pca, etiquetas)
		print("El mejor estimador encontrado para el modelo ", nombre_modelo, " usando RandomizedSearchCV es: ")
		print(grid_search.best_estimator_)
		mejores_estimadores_randomized_search[nombre_modelo] = grid_search.best_estimator_


	print("\n\nResultados buscando parametros con GridSearchCV: ")

	# entrenamos cada modelo con sus mejores parámetros con grid search
	for nombre_modelo, modelo in mejores_estimadores_grid_search.items():
		modelo_resultado, train_accuraccy, test_accuraccy = entrenar_modelo(modelo, predictores_pca, etiquetas)
		print("Accuraccy en train con ", nombre_modelo, ": ", train_accuraccy)
		print("Accuraccy en test con ", nombre_modelo, ": ", test_accuraccy)
		print()


	print("\n\n\nResultados buscando parametros con RandomizedSearchCV: ")

	# entrenamos cada modelo con sus mejores parámetros con grid search
	for nombre_modelo, modelo in mejores_estimadores_randomized_search.items():
		modelo_resultado, train_accuraccy, test_accuraccy = entrenar_modelo(modelo, predictores_pca, etiquetas)
		print("Accuraccy en train con ", nombre_modelo, ": ", train_accuraccy)
		print("Accuraccy en test con ", nombre_modelo, ": ", test_accuraccy)
		print()


if __name__ == "__main__":
	main()
