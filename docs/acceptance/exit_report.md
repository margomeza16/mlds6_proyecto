# Exit Report of Project <X> for Customer <Y>

Instructions: Template for exit criteria for data science projects. This is concise document that includes an overview of the entire project, including details of each stage and learning. If a section isn't applicable (e.g. project didn't include a ML model), simply mark that section as "Not applicable". Suggested length between 5-20 pages. Code should mostly be within code repository (not in this document).

Customer: <Enter Customer Name\>
	
	Juan Sebastian Lara
	
	Juan Sebastian Malagón

Team Members: <Enter team member' names. Please also enter relevant parties names, such as team lead, Account team, Business stakeholders, etc.\>

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

##	Solution Architecture
<Architecture of the solution, describe clearly whether this was actually implemented or a proposed architecture. Include diagram and relevant details for reproducing similar architecture. Include details of why this architecture was chosen versus other architectures that were considered, if relevant\>

##	Benefits
	
###	Company Benefit (internal only. Double check if you want to share this with your customer)
<What did our company gain from this engagement? ROI, revenue,  etc\>

###	Customer Benefit
What is the benefit (ROI, savings, productivity gains etc)  for the customer? If just POC, what is estimated ROI? If exact metrics are not available, why does it have impact for the customer?\>

##	Learnings

### 	Project Execution
<Learnings around the customer engagement process\>

### Data science / Engineering
<Learnings related to data science/engineering, tips/tricks, etc\>


### Domain
<Learnings around the business domain, \>

### Product
<Learnings around the products and services utilized in the solution \>

###	What's unique about this project, specific challenges
<Specific issues or setup, unique things, specific challenges that had to be addressed during the engagement and how that was accomplished\>

##	Links
<Links to published case studies, etc.; Link to git repository where all code sits\>
	
El siguiente es el link del repositorio del proyecto en GitHub:
https://github.com/margomeza16/mlds6_proyecto

##	Next Steps
 
<Next steps. These should include milestones for follow-ups and who 'owns' this action. E.g. Post- Proof of Concept check-in on status on 12/1/2016 by X, monthly check-in meeting by Y, etc.\>

## Appendix
<Other material that seems relevant – try to keep non-appendix to <20 pages but more details can be included in appendix if needed\>
