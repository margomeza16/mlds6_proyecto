# Exit Report of Project <X> for Customer <Y>

Instructions: Template for exit criteria for data science projects. This is concise document that includes an overview of the entire project, including details of each stage and learning. If a section isn't applicable (e.g. project didn't include a ML model), simply mark that section as "Not applicable". Suggested length between 5-20 pages. Code should mostly be within code repository (not in this document).

Customer: <Enter Customer Name\>
	
	Juan Sebastian Lara
	
	Juan Sebastian Malagón

Team Members: <Enter team member' names. Please also enter relevant parties names, such as team lead, Account team, Business stakeholders, etc.\>
	
	Alejandro Sandoval
	
	Marco Julio Gómez Amado

##	Overview

<Executive summary of entire solution, brief non-technical overview\>

La solución desarrollada consiste en la construcción y despliegue de un modelo de Deep Learning para realizar el diagnóstico automático de posibles enfermedades gastrointestinales y del colón, a partir, del estudio de imágenes de endoscopías digestivas y del colon.

##	Business Domain
<Industry, business domain of customer\>
	
Sector médico. Especificamente en la especialidad de Gastroenterología endoscópica. 

##	Business Problem
<Business problem and exact use case(s), why it matters\>
	
La detección automática de patologías medicas, mediante la aplicación de técnicas de machine learning, es de suma importancia, por cuanto permite contar con diagnósticos acertados, reduciendos costos económicos y tiempos en la obtención de los resultados, lo que se traduce en una mejor oportunidad de atención a los pacientes, ofreciéndoles un tratamiento más oportuno y acertado. Alineados con esta necesidad, se desarrollo este proyecto, en el que se construyó modelo de Deep Learning, para realizar el diagnóstico automático de posibles enfermedades del sistema digestivo y del colon.

##	Data Processing
<Schema of original datasets, how data was processed, final input data schema for model\>
	
EL conjunto de datos original esta integrado por imagenes endoscopicas con diagnósticos de enfermedades gastrointestinales o del colon, que de acuerdo a la fuente original (https://dl.acm.org/doi/pdf/10.1145/3083187.3083212), corresponden a imágenes reales, verificadas y etiquetadas por médicos especialistas en endoscopias. La clasificación de las imágenes corresponde a la patalogía médica encontrada, a saber:

/0_normal/: Corresponde a las imágenes con diagnostico normal (Sin enfermadad).

/1_ulcerative_colitis/: Corresponde a las imágenes con diagnóstico de colitis ulcerosa.

/2_polyps/: Corresponde a las imágenes con diagnóstico de pólipos.

/3_esophagitis/: Corresponde a las imágenes con diagnóstio de esofagitis.

Los imagenes se encuentran en formato .JPG, que según la fuente original ((https://dl.acm.org/doi/pdf/10.1145/3083187.3083212), tienen diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles y organizadas en carpetas nombradas de acuerdo con la patologia indicada antes. A su vez, se encuentran agrupadas en tres archivos .zip a saber:

train.zip. Archivo .zip Contiene un conjunto de 3200 imagenes de entrenamiento, separadas en 4 carpetas que corresponden a las clases de diagnóstico, indicadas antes.

test.zip. Archivo .zip Contiene un conjunto de 800 imagenes de prueba, separadas en 4 carpetas que corresponden a las clases de diagnóstico.

val.zip. Archivo .zip Contiene un conjunto de 2000 imagenes de validación, separadas en 4 carpetas que corresponden a las clases de diagnóstico.

## Preprocesamiento
	
 - Train.zip. Se aplicó el siguiente preprocesamiento:

Resizing. Al cargar las imágenes en arreglo de numpy, mediante la función tf.keras.preprocessing.image.load_img, se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224224 pixeles, con representación RGB. Este resize se realiza debido a que se implementará un modelo de Fine Tunning para clasificar las imagenes, utilizando la red convolucional ResNet50V2 para la extracción de características generales, la cual requiere que el tamaño de las imágenes de entrada al modelo sean d 224224 pixeles, con representación RGB. Este arreglo de numpy se almacena en la variable X_train, mientras que las etiquetas de clasifición de cada una de las imágenes, según la clase del diagnóstico al que pertenezcan, se almacenan en la variable y_train, asignándole alguno de los siguientes valores numéricos: 0 para diagnóstico normal, 1 para colitis ulcerativa, 2 para polipos y 3 para esofaguitis.

tf.keras.applications.resnet_v2.preprocess_input(X_train). Al conjunto X_train resultante del paso anterior se aplica el preprocesamiento de la red convolucional ResNet50V2, para escalar entre -1 y 1 los pixeles de las imágenes de entrada, que tienen un tamaño de 224*224 pixeles. Este preprocesamiento se debe aplicar a todas las imágenes que entren al modelo. La salida de este preprocesamiento se almacena en la variable X_train_prep.

Data augmentation. Debido a que las redes convolucionales requieren una gran cantidad de imágenes para que pueden aprender, al conjunto de entrenamiento resultante del paso anterior se aplica data augmentation, para generar nuevas imágenes transformadas (cambios en traslación, rotación, intensidad, entre otros) a partir del conjunto de imágenes original. De esta forma se amplia el tamaño del conjunto de imágenes de entrenamiento. La siguiente es la definición de las transformaciones realizadas al conjunto de entrenamiento:

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='constant')

Las imágenes generadas para el entrenamiento mediante las trasnformaciones quedan en la variable train_gen:

train_gen = train_datage.flow(X_train_prep, Y_train, batch_size=batch_size)

One-hot. Sobre la variable y_train, que contiene las etiquetas de clasificación del diagnóstico de cada imagen de entrenamiento, se aplica codificación one-hot para obtener una representación binaria de cada clase de diagnóstico de salida. El siguiente es el código aplicado: Y_train = tf.keras.utils.to_categorical(y_train).
	
- test.zip. Se realizó el siguiente preprocesamiento:
	
Resizing. Al cargar las imágenes en arreglo de numpy, mediante la función tf.keras.preprocessing.image.load_img, se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224224 pixeles, con representación RGB. Este resize se realiza debido a que se implementará un modelo de Fine Tunning para clasificar las imagenes, utilizando la red convolucional ResNet50V2 para la extracción de características generales, la cual requiere que el tamaño de las imágenes de entrada al modelo sean d 224224 pixeles, con representación RGB. Este arreglo de numpy se almacena en la variable X_test, mientras que las etiquetas de clasifición de cada una de las imágenes, según la clase del diagnóstico al que pertenezcan, se almacenan en la variable y_test, asignándole alguno de los siguientes valores numéricos: 0 para diagnóstico normal, 1 para colitis ulcerativa, 2 para polipos y 3 para esofaguitis.

tf.keras.applications.resnet_v2.preprocess_input(X_test). Al conjunto X_test, resultante del paso anterior, se aplica el preprocesamiento de la red convolucional ResNet50V2, para escalar entre -1 y 1 los pixeles de las imágenes de entrada, que tienen un tamaño de 224*224 pixeles. Este preprocesamiento se debe aplicar a todas las imágenes que entren al modelo. La salida de este preprocesamiento se almacena en la variable X_test_prep.

One-hot. Sobre la variable y_test, que contiene las etiquetas de clasificación del diagnóstico de cada imagen de prueba, se aplica codificación one-hot para obtener una representación binaria de cada clase de diagnóstico de salida. El siguiente es el código aplicado: Y_test= tf.keras.utils.to_categorical(y_test).

- val.zip. Se realizó el siguiente preprocesamiento:
	
Resizing. Al cargar las imágenes en arreglo de numpy, mediante la función tf.keras.preprocessing.image.load_img, se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224224 pixeles, con representación RGB. Este resize se realiza debido a que se implementará un modelo de Fine Tunning para clasificar las imagenes, utilizando la red convolucional ResNet50V2 para la extracción de características generales, la cual requiere que el tamaño de las imágenes de entrada al modelo sean d 224224 pixeles, con representación RGB. Este arreglo de numpy se almacena en la variable X_val, mientras que las etiquetas de clasifición de cada una de las imágenes, según la clase del diagnóstico al que pertenezcan, se almacenan en la variable y_val, asignándole alguno de los siguientes valores numéricos: 0 para diagnóstico normal, 1 para colitis ulcerativa, 2 para polipos y 3 para esofaguitis.

tf.keras.applications.resnet_v2.preprocess_input(X_val). Al conjunto X_val, resultante del paso anterior, se aplica el preprocesamiento de la red convolucional ResNet50V2, para escalar entre -1 y 1 los pixeles de las imágenes de entrada, que tienen un tamaño de 224*224 pixeles. Este preprocesamiento se debe aplicar a todas las imágenes que entren al modelo. La salida de este preprocesamiento se almacena en la variable X_val_prep.

One-hot. Sobre la variable y_val, que contiene las etiquetas de clasificación del diagnóstico de cada imagen de validación, se aplica codificación one-hot para obtener una representación binaria de cada clase de diagnóstico de salida. El siguiente es el código aplicado: Y_test= tf.keras.utils.to_categorical(y_val).
	
	
##	Modeling, Validation
<Modeling techniques used, validation results, details of how validation conducted\>
	
Se construyo un modelo de Fine Tunning, partiendo de la red convolucional preentrenada ResNet50V2 para la extracción de características, sobre esta se adicionaron 4 capas finales para la clasificación. El siguiente enlace muestra la arquitectura de la red de fine tunning construida:
	
https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/acceptance/arq_ft.jpg

No se incluyeron las capas finales de la ResNet50V2, en su lugar se adicionaron las siguientes 4 capas de clasificación:
	
#Capa de global average pooling

pool = tf.keras.layers.GlobalAveragePooling2D()(extractor.output)

#capa densa con 32 neuronas y activación relu

dense1 = tf.keras.layers.Dense(32, activation="relu")(pool)

#Capa de dropout con taza de 0.2 para regularización

drop1 = tf.keras.layers.Dropout(0.2)(dense1)

#Capa densa de salida con 4 clases con activación softmax

dense2 = tf.keras.layers.Dense(4, activation="softmax")(drop1)
	
El modelo se compilo con los siguientes parámetros:
	
ft_model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-3),
                 metrics=["accuracy"])

Se hizo un precalentamiento de dos epocas con los siguientes parámetros:

batch_size = 32
	
ft_model.fit_generator(train_gen, validation_data=(X_val_prepro, Y_val),
                       epochs=2, steps_per_epoch=X_train.shape[0]//batch_size)

Después del precalentamiento, se compila el modelo con los siguientes parámetros:
	
ft_model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-4),
                 metrics=["accuracy"])
	

El entrenamiento del modelo se hizo con los siguientes parámetros:

Se descongelaron para entrenar las últimas 13 capas convolucionales de la red ResNet50V2 más las 4 capas de clasificación adicionadas al final.

batch_size = 32
	
hist_ft = ft_model.fit(train_gen, validation_data=(X_val_prepro, Y_val),
                                 epochs=20, steps_per_epoch=X_train.shape[0]//batch_size,
                                 callbacks=[best_callback])

Para la validación del modelo se utilizaron las siguientes metricas:

Comparación de las curvas de pérdida y validación del conjunto de imágenes de test vs el conjunto de imágenes de validación. La siguiente gráfica muestra estas curvas:

![](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/acceptance/curvas_ft.jpg)
	
	
Evaluación en terminos de accuracy, precision, recall y f1. Obteniendo los siguientes resultados:
	
![](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/modeling/metricas%20ft.jpg)
	
##	Solution Architecture
<Architecture of the solution, describe clearly whether this was actually implemented or a proposed architecture. Include diagram and relevant details for reproducing similar architecture. Include details of why this architecture was chosen versus other architectures that were considered, if relevant\>
	
Se utilizó como base la metodología de ciencia de datos ágil e iterativa Team Data Science Process – TDSP, por lo que se trató de una arquitectura de solución implementada. En el repositorio GIT *mlds6_proyecto* se estructuraron cada uno de los componentes de las ramas (*branchs*) inicialmente sobre la rama de desarrollo (*dev*) y posteriormente en la rama principal (*main*). La solución requirió del uso de: la librería Poetry como gestor de paquetes y solucionador de dependencias, la imagen de un contenedor (Docker) específicamente el de tensorflow-tensorflow utilizado como ambiente virtual y la librería de *MLFlow* para el despliegue del modelo de la red neuronal convolucional entrenada. La librería construida con el gestor *Poetry* se denominó *endos_images*, y constó de los módulos que se describen a continuación: 
	
•	Data.py: En este módulo se estructuraron las clases *DataLoader* y *DataLoaderAug* encargadas de cargar las imágenes de entrenamiento, prueba y validación extraídas del entorno de la nube de GoogleDrive.
•	Models.py: En este módulo se estructuraron las clases de *MyModel*, *CompileHParams*, *TrainHParams* y *ModelBuilder* encargadas de la construcción de la arquitectura, el compilado y los parámetros modelo de la red neuronal convolucional.
•	Train.py: En este módulo se estructuraron las funciones de *make_parser* y del *main* para construir, compilar, entrenar y guardar los artefactos correspondientes al modelo en sí, las métricas y las gráficas de pérdidas y exactitud generadas por el modelo a partir del orquestador de *MLFlow*.
•	Viz.py: En este módulo se estructuro la clase de *Visualizer* como generadora de las gráficas de las pérdidas y métrica de exactitud del modelo.
	
En la siguiente figura se detalla la arquitectura de la solución implementada:

![DataFlow](https://user-images.githubusercontent.com/73256719/206821631-76d1b12c-4950-4f66-9a3a-2117451bacba.png)

##	Benefits
	
###	Company Benefit (internal only. Double check if you want to share this with your customer)
<What did our company gain from this engagement? ROI, revenue,  etc\>
	
El beneficio interno de nuestra empresa corresponde a generar un Goodwill en el sector de tecnologías relacionadas al sector de salud mediante la implementación de soluciones que permitan automatizar la obtención de diagnósticos de patologías como lo son las gastrointestinales y del colón. A partir de incrementar el goodwill nuestra empresa se posicionará mejor en el mercado generando mayores ingresos.

###	Customer Benefit
What is the benefit (ROI, savings, productivity gains etc)  for the customer? If just POC, what is estimated ROI? If exact metrics are not available, why does it have impact for the customer?\>
	
Se espera que a partir de la automatización de generación de diagnósticos el cliente se beneficie en la mejora en los tiempos de entrega. Esto se ve reflejado en disminuir el tiempo de esta etapa y avanzar en las post diagnóstico para la atención y tratamiento de las patologías gastrointestinales y del colón.

##	Learnings

### 	Project Execution
<Learnings around the customer engagement process\>
	
El aprendizaje sobre el proceso de interacción del cliente fue limitado ya que al tratarse de un ejercicio académico el cliente fue hipotético y no se pudo profundizar en como se habría mejorado la solución implementada.

### Data science / Engineering
<Learnings related to data science/engineering, tips/tricks, etc\>

Los conocimientos adquiridos en la ejecución de este proyecto fueron bastante amplios y satisfactorios para nuestro grupo. Podemos iniciar nombrando que se intento replicar lo mejor posible la metodología de Team Data Science Process – TDSP delegando roles y actividades especificas a cada uno de los miembros del equipo. A partir de esto, adquirimos habilidades y destrezas relacionadas a Machine Learning Operations MLOPS buscando agilizar los procesos de Machine Learning y en nuestro caso Deep Learning, mediante el uso de librerías y artefactos de Python y Linux que desconocíamos al inicio de este módulo tales como: Docker, MLFlow y Poetry. Finalmente, y no menos importante, reconocimos la importancia de estructurar modularmente el código para su respectiva reproducción por otras personas mediante el uso de la programación orientada a objetos.

### Domain
<Learnings around the business domain, \>
	
El aprendizaje en torno al dominio empresarial principalmente fue adquirir los conocimientos y practicidad de las metodologías de TDSP y MLOPS para el desarrollo de soluciones que se encuentren debidamente documentadas y estructuradas de tal forma que sean fáciles de persistir y mantener dada la estructuración modular del código.

### Product
<Learnings around the products and services utilized in the solution \>
	
Son bastantes los aprendizajes adquiridos en cuanto a los productos y servicios implementados en la solución del proyecto, pero, en el equipo destacamos principalmente la herramienta de MLFlow como gestor en el despliegue de modelos que al inició de este curso desconocíamos y que representa una herramienta imperativamente lucrativa que podremos utilizar en nuestro ámbito laboral. 

###	What's unique about this project, specific challenges
<Specific issues or setup, unique things, specific challenges that had to be addressed during the engagement and how that was accomplished\>
	
Fueron bastantes los desafíos que tuvimos que afrontar como equipo para poder implementar la solución planteada inicialmente, sin embargo, nos gustaría principalmente destacar cuatro (4):
	
1)	El equipo carecía de conocimientos en programación orientada a objetos – POO, por lo que fue un gran desafío estructurar y modular el código en clases y métodos.
2)	La configuración del contendor (Docker) represento otro obstáculo a superar dado que teníamos conocimientos básicos en el área.
3)	La carga de datos para el entrenamiento del modelo fue un reto dada la volumetría de los datos.
4)	El despliegue del modelo representó un desafío ya que ninguno de los miembros del equipo contaba con conocimientos HTLM, por lo que desconocíamos el protocolo de peticiones (requests).


##	Links
<Links to published case studies, etc.; Link to git repository where all code sits\>
	
El siguiente es el link del repositorio del proyecto en GitHub:
https://github.com/margomeza16/mlds6_proyecto

##	Next Steps
 
<Next steps. These should include milestones for follow-ups and who 'owns' this action. E.g. Post- Proof of Concept check-in on status on 12/1/2016 by X, monthly check-in meeting by Y, etc.\>
	
Se considera que los siguientes pasos de este proyecto pueden ser: 1) Ampliar las clases de patologías gastrointestinales y del colón para generalizar aún más la solución en el sector de salud y 2) Potenciar la solución con sugerencias de tratamientos a la patología diagnosticada.

## Appendix
<Other material that seems relevant – try to keep non-appendix to <20 pages but more details can be included in appendix if needed\>

Para el desarrollo de este proyecto se tomaron las siguiente fuentes:
 - Las imágenes de endoscopías para el entrenamiento, prueba y validación se descargaron de Kaggle, de la siguiente dirección:
	
https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning?resource=download

Como soporte médico para la motivación de este proyecto, se tomaron en cuenta los siguientes paper's, desde el link de Kaggle, en los cuales se detalle el trabajo realizado para la obtención de las imágenes diagnóstico y el propósito de disponer de las mismas para implementar soluciones automatizadas mediante técnicas de machine learning que permitan hacer el diagnóstico de patalogías gastrointestinales y del colon de forma más rápida, menos costosa y con un nivel alto de confianza. A continuación se transcriben textualmente los enlaces:
	
KVASIR Dataset (https://dl.acm.org/doi/abs/10.1145/3083187.3083212)
K. Pogorelov et al., "KVASIR", Proceedings of the 8th ACM on Multimedia Systems Conference, 2017. DOI: 10.1145/3083187.3083212.

ETIS-Larib-Polyp DB Dataset (https://link.springer.com/article/10.1007/s11548-013-0926-3/)
J. Silva, A. Histace, O. Romain, X. Dray and B. Granado, "Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer", International Journal of Computer Assisted Radiology and Surgery, vol. 9, no. 2, pp. 283-293, 2013. DOI:10.1007/s11548-013-0926-3.
	
This is used in the study:

F. J. P. Montalbo, "Diagnosing Gastrointestinal Diseases from Endoscopy Images through a Multi-Fused CNN with Auxiliary Layers, Alpha Dropouts, and a Fusion Residual Block," Biomedical Signal Processing and Control (BSPC), vol. 76, July, 2022, doi: 10.1016/j.bspc.2022.103683

Paper link: https://www.sciencedirect.com/science/article/pii/S1746809422002051
	
