# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach
* What is target definition

El objetivo es construir un modelo de Machine Learning para realizar la predicción automática del diagnóstico de posibles enfermedades gastrointestinales y del colon a partir del análisis de una imagen de endoscopía digestiva. Se consideran 4 posibles clases de diagnóstico a saber:

0: Diagnostico normal (Sin enfermadad).

1: Diagnóstico de colitis ulcerosa.

2: Diagnóstico de pólipos.

3: Diagnóstio de esofagitis.

* What are inputs (description)

Las entradas corresponden a imágenes de endoscopias del esofago y del colon. Los imágenes se encuentran en formato .JPG, que según la fuente original (https://dl.acm.org/doi/pdf/10.1145/3083187.3083212), fueron tomadas por especialistas en gastroenterología endoscópica a pacientes reales, tienen diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles. Se encuentran organizadas en carpetas nombradas de acuerdo con la patologia indicada antes. En total se tienen 4200 imágenes, distribuidas de la siguiente forma: 

3200 para entrenamiento. Separadas en 4 carpetas según el diagnóstio, cada uno con 800 imágenes.

800 para test. Separadas en 4 carpetas según el diagnóstio, cada uno con 200 imágenes.

2000 para validación. Separadas en 4 carpetas según el diagnóstio, cada uno con 500 imágenes.

* What kind of model was built?

Se construyó un modelo de transfer learning y uno de fine tuning, utilizando redes convoluciones preentrenadas. Concretamente se utilizo la red ResNet50V2 para la extracción de características, sobre la cual se adicionaron capas densas para la clasificación.


## Model Description

* Models and Parameters

	* Description or images of data flow graph
  		* if AzureML, link to:
    		* Training experiment
    		* Scoring workflow
	* What learner(s) were used?
	* Learner hyper-parameters

Como se indico en el numeral anterior, para la construcción de los modelos de transfer learning y fine tunning, se partió de la red convolucional ResNet50V2 como modelo base para la extracción de carterísticas básicas de las imágenes. Se tomó la red convolucional sin las capas densas finales (TOP) y se adicionaron las siguientes capas densas para la clasificación:

#Capa de global average pooling

pool = tf.keras.layers.GlobalAveragePooling2D()(extractor.output)

#capa densa con 32 neuronas y activación relu

dense1 = tf.keras.layers.Dense(32, activation="relu")(pool)

#Capa de dropout con taza de 0.2 para regularización

drop1 = tf.keras.layers.Dropout(0.2)(dense1)

#Capa densa de salida con 4 clases con activación softmax

dense2 = tf.keras.layers.Dense(4, activation="softmax")(drop1)

Los parámetros utilizados para transfer learning son los siguientes:

 batch_size = 32.
 
 optimizador Adam con learning rate = 1e-3.
 
 epocas=20
 
 Función de perdida: "categorical_crossentropy" .
 
 metrics = ["accuracy"].

Los parámetros para fine tunning son los siguientes:

batch_size = 32.
 
 optimizador Adam con learning rate = 1e-4.
 
 epocas=20
 
 Función de perdida: "categorical_crossentropy" .
 
 metrics = ["accuracy"].

El precalentamiento para fine tunning, se hizo con 2 épocas y el mismo learning rate utilizado para transfer learning (learning rate = 1e-3).

Este modelo de fine tuning, con los anteriores parámetros, fue seleccionado como el mejor, por las métricas de desempeño btenidas, como son, accuracy, precision, F1-Score, recall y las curvas de pérdida y validación  en entrenamiento y validación. Se llego a este modelo después de probar con diferentes valores de hiperparámetros como learning rate, batch_size, epocas, cambiando el número de neuronas del capa densa y adicionando más capas densas al final del modelo. Siendo éste, con el que se lograron obtener las mejores metricas.

Asi mismo, por sus resultados, consumo de recursos y tiempo de entrenamiento, se selecciono la red ResNet50V2, después de ensayar con varios modelos pre-entrenados, como MobileNetV2 y DenseNet12.

## Results (Model Performance)
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable

A continuación se muestra la comparación de las curvas de accuracy y pérdida de los modeles de transfer learning y fine tuning:

https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/modeling/rendimiento%201.jpg

## Model Understanding

* Variable Importance (significance)

* Insight Derived from the Model

## Conclusion and Discussions for Next Steps

* Conclusion on Feasibility Assessment of the Machine Learning Task

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
