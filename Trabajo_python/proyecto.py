# -*- coding: utf-8 -*-

#
# Bibliotecas que utilizaremos
#

import os
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

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

### para ignorar los warnings de convergencia de sklearn
### ya sabemos que faltan iteraciones para una red neuronal, pero ya tarda
### bastante con toda la busqueda de hiperparámetros
import warnings
import sklearn.exceptions
warnings.filterwarnings(action='ignore', category=sklearn.exceptions.ConvergenceWarning)

#
# Constantes
#

DIR_MODELOS = "modelos/"
NUM_CPUS = 4
GRID_SEARCH_EXTENSION = "_GridSearchCV.pck"
RANDOMIZED_SEARCH_EXTENSION = "_RandomizedSearchCV.pck"

#
# Funciones
#

def pausa():
	"""
	Para realizar una pausa en el código y revisar los resultados que se muestran
	en pantall
	"""
	input("\n\n--- Pulsa una tecla para continuar ---\n\n")


def normalizar_datos(datos):
	"""
	Función para normalizar unos datos dados utilziando un escalado
	estandar de scikit learn

	Recibe:
		datos: Datos a escalar

	Devuelve:
		Datos escalados utilizando sklearn.preprocessing.StandardScaler
	"""
	escalado = skl.preprocessing.StandardScaler()
	escalado.fit(datos)
	return escalado.transform(datos)

def almacenar_modelos(modelos, extension, ruta = DIR_MODELOS):
	"""
	Función para almacenar en disco los modelos de scikit learn con pickle.
	Como nombre del fichero se utilizará el nombre del modelo añadiendo la
	extensión dada como parámetro.

	Recibe:
		modelos: Lista de modelos de scikit-learn a almacenar
		extension: Extensión a añadir al almacenar los modelos
		ruta: Ruta donde almacenar los modelos

	"""

	# para cada modelo, le añado la extensión y hago el dump con pickle
	for estimador in modelos:
		nombre_fichero = type(estimador).__name__ + extension
		with open(ruta + "/" + nombre_fichero , "wb") as f:
			pickle.dump(estimador, f)


def cargar_modelos(modelos, ruta = DIR_MODELOS):
	"""
	Función para cargar los modelos dados desde la ruta dada

	Recibe:
		modelos: Lista de modelos, con los que se obtendrá el nombre para recuperarlos
		ruta: Ruta donde se encuentran almacenados los modelos
	"""

	i = 0
	he_podido_cargar_modelos = True

	# creo los diccionarios vacios
	mejores_estimadores_grid_search = dict()
	mejores_estimadores_randomized_search = dict()

	# ciclo mientras no encuentre un error
	while he_podido_cargar_modelos and i < len(modelos):
		# creo los nombres de los modelos a cargar
		nombre_modelo = type(modelos[i]).__name__
		nombre_modelo_grid = nombre_modelo + GRID_SEARCH_EXTENSION
		nombre_modelo_randomized = nombre_modelo + RANDOMIZED_SEARCH_EXTENSION

		# miro que existan en disco
		he_podido_cargar_modelos = os.path.exists(ruta + "/" + nombre_modelo_grid) and os.path.exists(ruta + "/" + nombre_modelo_randomized)

		# si existen, los cargo con pickle
		if he_podido_cargar_modelos:
			with open(ruta + "/" + nombre_modelo_grid, "rb") as f:
				mejores_estimadores_grid_search[nombre_modelo] = pickle.load(f)

			with open(ruta + "/" + nombre_modelo_randomized, "rb") as f:
				mejores_estimadores_randomized_search[nombre_modelo] = pickle.load(f)

		i += 1

	# si no he podido cargarlos, devuelvo None
	if not he_podido_cargar_modelos:
		mejores_estimadores_grid_search = None
		mejores_estimadores_randomized_search = None

	return mejores_estimadores_grid_search, mejores_estimadores_randomized_search


def lista_modelos_a_usar():
	"""
	Función que nos devuelve los modelos que utilizaremos
	Esta función existe solo para especificar una unica vez que modelos utilizaremos
	"""

	# todos los modelos que lo permiten trabajarán con NUM_CPUS para aplicar paralelismo y
	# realizar el proceso de busqueda de parámetros más rápido
	modelos = [skl.linear_model.LogisticRegression(),
			   skl.tree.DecisionTreeClassifier(),
			   skl.ensemble.RandomForestClassifier(n_jobs = NUM_CPUS),
			   skl.svm.SVC(),
			   skl.neural_network.MLPClassifier()]

	return modelos


def grid_parametros_a_usar():
	"""
	Función que nos devuelve los hiperparámetros donde buscaremos los mejores modelos
	Esta función existe solo para especificar una unica vez que hiperparámetros utilizaremos
	"""

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


	parametros["MLPClassifier"] = {"activation": ["relu", "tanh", "logistic"],
								   "solver": ["adam", "sgd"],
								   "alpha": [0.0001, 0.0005, 0.001, 0.005],
								   "learning_rate": ["constant", "invscaling", "adaptive"],
								   "max_iter" : [300, 400, 500, 600]}

	return parametros

def busqueda_hiperparametros(modelos, parametros, X, Y, funcion_busqueda):
	"""
	Función para realizar una busqueda de hiperparámetros en una lista de modelos
	dada, con cierta función de busqueda

	Recibe:
		modelos: Modelos donde realizar la búsqueda
		parametros: Diccionario con los nombres de modelos como clave y un diccionario
			con el grid de parámetros donde buscaremos como valor.
		X: Predictores con los que buscar los mejores parámetros
		Y: Etiquetas con los que buscar los mejores parámetros.
		funcion_busqueda: Función de sklearn.model_selection con la que realizar
			la búsqueda de hiperparámetros
	"""

	mejores_estimadores = dict()

	for modelo in modelos:
		nombre_modelo = type(modelo).__name__
		grid_search = funcion_busqueda(modelo, parametros[nombre_modelo])
		grid_search.fit(X, Y)
		print("El mejor estimador encontrado para el modelo ", nombre_modelo, " es: ")
		print(grid_search.best_estimator_)
		print()
		mejores_estimadores[nombre_modelo] = grid_search.best_estimator_

	return mejores_estimadores


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


	return resultado["estimator"][mejor_modelo], train_accuraccy_cv, test_accuraccy





def main():
	# leemos los datos
	datos = pd.read_csv("datos/SouthGermanCredit.csv", sep = " ")
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

	pausa()

	# escalamos los datos antes de aplicar el PCA, ya que PCA calculará unos nuevos
	# predictores a partir de los actuales, y si no están escalados le dará más
	# importancia a unos que a otros
	# utilizamos un standard scaler para normalizar (media 0 y desviación 1)
	predictores_escalados = normalizar_datos(predictores)

	print("Miramos los predictores tras escalarlos: ")
	print(predictores_escalados[0:5])

	pausa()

	# aplicamos PCA, dejando tantas características como sean necesarias
	# para explicar un 90% de los datos
	modelo_pca = skl.decomposition.PCA(n_components = 0.9)
	modelo_pca.fit(predictores_escalados)
	# como vemos con una nos basta, hemos pasado de 20 predictores a 1
	print("Porcentajes de varianza explicados por cada atributo resultante de PCA final: ")
	print(modelo_pca.explained_variance_ratio_)

	predictores_pca = modelo_pca.transform(predictores_escalados)

	print("Tamaño de los predictores tras PCA: ", predictores_pca.shape)

	pausa()

	modelos = lista_modelos_a_usar()

	# parametros para los modelos
	parametros = grid_parametros_a_usar()

	mejores_estimadores_grid_search = dict()
	mejores_estimadores_randomized_search = dict()

	if not os.path.exists(DIR_MODELOS):
	 	os.mkdir(DIR_MODELOS)


	he_podido_cargar_modelos = False

	# miro si tengo que cargar los modelos
	if "cargar_modelos" in sys.argv:
		print("Cargando modelos de la carpeta: ", DIR_MODELOS)

		mejores_estimadores_grid_search, mejores_estimadores_randomized_search = cargar_modelos(modelos)
		he_podido_cargar_modelos = mejores_estimadores_grid_search is not None



	# si no los he podido cargar (ya sea porque no quería, o por un error)
	# los buscamos
	if not he_podido_cargar_modelos:
		print("\nPasamos a buscar los mejores parámetros para cada modelo.")

		mejores_estimadores_grid_search = busqueda_hiperparametros(modelos,
																   parametros,
																   predictores_pca,
																   etiquetas,
																   skl.model_selection.GridSearchCV)

		pausa()

		mejores_estimadores_randomized_search = busqueda_hiperparametros(modelos,
																		 parametros,
																		 predictores_pca,
																		 etiquetas,
																		 skl.model_selection.RandomizedSearchCV)

	# guardamos los modelos si no los hemos podido cargar
	if not he_podido_cargar_modelos:
		almacenar_modelos(mejores_estimadores_grid_search, GRID_SEARCH_EXTENSION)
		almacenar_modelos(mejores_estimadores_randomized_search, RANDOMIZED_SEARCH_EXTENSION)


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
