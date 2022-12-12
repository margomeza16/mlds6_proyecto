# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach
* What is target definition

El objetivo es construir un modelo de Deep Learning para realizar la predicción automática del diagnóstico de posibles enfermedades gastrointestinales y del colon a partir del análisis de una imagen de endoscopía digestiva. Se consideran 4 posibles clases de diagnóstico a saber:

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

Asi mismo, por sus resultados, tamaño, consumo de recursos y tiempo de entrenamiento, se selecciono la red ResNet50V2, después de ensayar con varios modelos pre-entrenados, como MobileNetV2 y DenseNet12.

## Results (Model Performance)
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable

A continuación se muestra la comparación de las curvas de pérdida (imágen de la izquierda) y accuracy (imágen de la derecha) de los modeles de transfer learning y fine tuning:
![](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/modeling/rendimiento%201.jpg)

De las anteriores gráficas comparativas, se observa que el modelo de fine tuning tiene, ligeramente, un mejor desempeño que transfer learning, ya que las curvas de perdida para entrenamiento y validación, son muy similares en su forma y cercanas, se mantienen paralelas y sin grandes variaciones en su trayectoria, con tendencia a la baja, desde las primeras iteraciones y estan un poco por debajo de las cuervas de perdida de transfer learning. Asi mismo, las curvas de accuracy para entrenamiento y validación, tambien tienen una forma similar, sin grandes variaciones en su trayectoria, paralelas y cercanas, con valores altos desde el comienzo de las iteraciones y con tendencia al alza a lo largo de las iteraciones y se mantienen por arriba de las curvas de transfer learning. Por su parte, las curvas de perdida de transfer learning, tanto para entrenamiento como para validación tambien tienden a la baja a lo largo de las iteraciones, solo que su forma es más irregular (presentan grandes variaciones en su trayectoria) y estan un poco por arriba de las curvas de fine tuning. De forma similar las curvas de accuracy de transfer learning, tanto para entrenamiento como para validación, tambien tienden al alza a lo largo de las iteraciones, pero con irregularidades (variaciones) en su trayectoria y estan un poco por debajo del accuracy de fine tuning.

A continuación se muestran las métricas de desempeño de Transfer Learning y Fine Tunning:
![](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/modeling/metricas%20tl.jpg)

![](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/modeling/metricas%20ft.jpg)

Aunque en terminos generales las anteriores métricas de los dos modelos son muy buenas, con valores promedios de precision, recall y f1-score, muy similares, al revisarlas en detalle, se observar que el accuracy y las medias de precision, recall y f1-score son un poco mejores en el modelo de fine tuning en la mayoria de las clases, lo que se puede interpretar como que logra, ligeramente, un mejor desempeño en la clasificación de las imagenes en las 4 clases.

## Model Understanding

* Variable Importance (significance)

* Insight Derived from the Model

Teniento en cuenta las interpretaciones de las anteriores métricas, tanto el modelo de Transfer Learning como el modelo de Fine Tunning, construidos, lograron una excelente desempeño. Estos resultados se lograron gracias al uso de una red convolucional entrenada como modelo base y robusto para la extracción de características, gracias a su entrenamiento con millones de imágenes. Asi mismo, facilitó la construcción de estos modelos de transfer learning y fine tunning, evidenciando un resultado más preciso en la clasificación de las imágenes.

Con base a las anteriores interpretaciones, basadas en la comparación de los dos modelos en terminos de las metricas de precision, recall, f1-score y accuracy y de las curvas de perdida y acuracy en entrenamiento y validación, se concluye que el modelo de Fine Tuning da un mejor resultado. Resaltando los valores promedios de precision, recall y f1-score del mejor modelo de fine tuning, que alcanzan el valor del 99%, frente al 98% obtenido por el mejor modelo de transfer learning, que tambien es muy bueno. Asi mismo, se observa que el accuracy y las medias de precision, recall y f1-score son un poco mejores en el modelo de fine tuning para cada una de las clases, lo que se puede interpretar como que logra, ligeramente, un mejor desempeño en la clasificación de las imagenes en las 4 clases. De igual manera, como se explico en el numeral anterior, la forma de las curvas de perdida y accuracy de entrenamiento y validación, su tendencia, cercania, similitud y variación, desde el comienzo y a lo largo de las iteraciones son mejores en Fine Tuning.

## Conclusion and Discussions for Next Steps

* Conclusion on Feasibility Assessment of the Machine Learning Task

Gracias a los resultados obtenidos, el modelo desarrollo en este proyecto es de gran aplicación, ya que permite lograr diagnósticos muy acertados, ahorrando tiempo y costos, lo cual se deriva en una mejor atención a los pacientes, permitiéndoles la oportunidad de contar con tratamiento más rápido y acertado, con base al diagnóstico emitido por el modelo.

Un aprendizaje del desarrollo de este proyecto,  es que se tiene una responsabilidad muy grande al desarrollar modelos relacionados con el diagnóstico de posibles enfermedades a partir de imágenes médicas, ya que antes de su puesta en producción, se deben someter a muchas pruebas y tuning para lograr unas metricas de precisión muy altas y similares en cada una de las clases y en los valores promedios (superiores al 99%), ya que las consecuencias de predicciones erroneas, puede ser muy alto, al tratarse de vidas humanas.

Como siguientes pasos, se considera la posibilidad de ampliar las clases de diagnóstico del modelo desarrollo, de tal por forma que se pueda ampliar su uso. Para lograrlo es necesario, contar con una cantidad considerable de imágenes de endoscopias digestivas asociadas a otras patalogías.

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

La data actual, que son las imágenes etiquetadas según su diagnóstico, por especialistas, podría ser enriquecido con imágenes de nuevos diagnósticos, tanto de las clases actuales, como de otro tipos de patologías. Esto permitiría reentrenar el modelo y ampliar su uso, al incluir nuevos tipos de diagnósticos.

* What other Relevant Data Sources Are Available to Help the Modeling

Para ayudar a enriquecer el modelo, es necesario contar con base pública de imágenes de endoscopias digestivas, clasificadas por su diagnóstico, de las diferentes instituciones médicas que realicen este tipo de exámenes.
