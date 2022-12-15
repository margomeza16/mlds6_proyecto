# Deployment

In this folder you can add deployment documentation, including

* Documentation for the APIs (e.g. swagger).

No Aplica.

* Package documentation (e.g. sphinx).

La aplicación fue construida utilizando las siguientes librerías:

Poetry: Para la instalación y manejo de dependencias.

mlflow: Para la estructuración y despliegue del proyecto.


**Modft** es una biblioteca para el lenguaje de programación Python que permite predecir la probabilidad de una patología gastrointestinal de un paciente dada una endoscopia mediante el uso de una red neuronal convolucional. En su versión 0.1.0.

**modft** fue construida a partir de las siguientes librerías base: tensorflow, mlflow, numpy, matplotlib, os, zipfile, dataclasses, typing y argparse.  Esta librería es soportada por las versiones de Python mayores o iguales a 3.10 y menores que 3.11, requirió de la dependencia de desarrollo de neovim en su versión 0.3.1, además de las siguientes dependencias: 

*   **docker-compose** en su versión 1.29.2.

*   **mlflow** en su versión 2.0.1.

*   **tensorflow** en su versión 2.11.0.

modft cuenta con los siguientes módulos que se describen a continuación:

•	**modft.data**: Carga y preprocesamiento de los datos de entrada del modelo.

•	**modft.models**: Construcción y compilado del modelo de red neuronal convolucional. 

•	**modft.viz**: Visualización del rendimiento del modelo.

En el módulo *data* se encuentra contenida la siguiente clase DataLoader encargada de realizar el cargue y preprocesamiento de los datos correspondientes a imágenes RGB provenientes de un repositorio en GoogleDrive retornando el conjunto de datos:

*train_gen*, *X_train_prep*, *X_val_prep*, *X_test_prep*, *Y_train*, *Y_test*, *Y_val* en arreglos de Numpy de un tamaño de (224,224,3). 

Además , la clase *DataLoader* contiene las funciones de:

•	**init**: Constructor de la clase *DataLoader*: Parámetro self: objeto de la clase *DataLoader*.

•	**load_file**(orig, dest): Extractor de los archivos compresos en el repositorio de GoogleDrive. Parámetro *orig*: string que indica la ruta de origen. Parámetro *dest*: string que indica la ruta de destino.

•	**preproceso**(ruta_dest): Preprocesamiento de las imágenes descomprimidas en la ruta destino retornando las salidas en arreglos de numpy de (224,224,3). Parámetro *ruta_dest*: string que indica la ruta de destino donde fueron descomprimidos los datos.

•	**call**: Llamado de la clase DataLoader que retorna el conjunto de datos: train_gen, X_train_prep, X_val_prep, X_test_prep, Y_train,Y_test, Y_val en arreglos de Numpy de un tamaño de (224,224,3). Parámetro self: objeto de la clase *DataLoader*.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------

En el módulo models se encuentran contenidas las clases de *MLP* y *ModelBuilder*, la primera encargada de definir la arquitectura de la red neuronal convolucional y la segunda encargada de construir, compilar y entrenar el modelo retornando un objeto de keras de tipo keras.engine.functional.Functional y un objeto de tipo *ModelBuilder*. 

Además , la clase **MLP** contiene las funciones de:

•	**init**: Constructor de la clase MLP: Parámetro self: objeto de la clase *MLP*, *dropout*: float indica el valor de dropout del modelo, *args y **kwargs objetos derivados.

•	**call**: Llamado de la clase *MLP*. Parámetro self: objeto de la clase *MLP*, *x*: tensor entradas del modelo.

Por otra parte, la clase *ModelBuilder* contiene las funciones de: 

•	**set_compile_hparams**: Configurador de los parámetros de compilado del modelo: Parámetro *self*: objeto de la clase ModelBuilder. *Compile_hparams*: objeto de la clase de datos de *Compile_hparams*. Retorna un objeto de la clase *ModelBuilder*.

•	**set_train_hparams**: Configurador de los parámetros de entrenamiento del modelo: Parámetro *self*: objeto de la clase ModelBuilder. *Train_hparams*: objeto de la clase de datos de *Train_hparams*. Retorna un objeto de la clase *ModelBuilder*.

•	**build**: Constructor y compilador del modelo: Parámetro *self*: objeto de la clase *ModelBuilder*. Retorna un objeto de la clase *ModelBuilder*.

•	**train**: Entrenador del modelo: Parámetro *self*: objeto de la clase *ModelBuilder*. *Dl*: objeto de la clase *DataLoader*. Retorna un objeto de la clase *ModelBuilder*.


----------------------------------------------------------------------------------------------------------------------------------------------------------------------


En el módulo **viz** se encuentra contenida la clase de Visualizer encargada de proporcionar un gráfico del rendimiento del modelo. Además, contiene las funciones de:
Además , la clase *MLP* contiene las funciones de:

•	**init**: Constructor de la clase *Visualizer*: Parámetro *self*: objeto de la clase *Visualizer*, *history*: Dict contiene las métricas de perdida y accuracy del modelo, *path*: string contiene la ruta de visualización de la imagen.

•	**Call**: Llamado de la clase *Visualizer*. Parámetro *self*: objeto de la clase *Visualizer*, proporciona el gráfico de las pérdidas y exactitud del modelo.

Adicionalmente, **modft** cuenta con los siguientes scripts: 

•	**modft.train**: Construcción, compilación, entrenamiento y visualización del rendimiento del modelo mediante el uso del API de MLFlow.

•	**modft.test**: Carga del modelo, de los datos de prueba y predicción de la probabilidad de la patología. Se utiliza para recibir una imagen del usuario y realizar el diagnóstico de la patalogía.

El script de train contiene las funciones de:

•	**make_parser**: Retorna argumentos de tipo ArgumentParser para los parámetros del entrenamiento del modelo de: dropout (float), learning_rate (float), epochs (int) y bath_size (int).

•	**main**: Contiene los procedimientos de construcción, compilado, entrenamiento y visualización del rendimiento del modelo a partir de los objetos DataLoader, Compile_hparams, Train_hparams, ModelBuilder y Visualizer orquestados por mlflow.


El script de test contiene las funciones de:

•	**main**: Contiene los procedimientos de cargue de los datos de prueba, del modelo entrenado y predicción de la probabilidad de la patología a partir del objeto DataLoader orquestado por mlflow.


* Dashboard documentation.

No Aplica.

* Any other documentation depending on the deployment kind.

Para la estructuración, instalación y manejo de dependencias del proyecto se utilizó poetry, destacando la siguiente secuencia de comandos principales:


poetry ini. Para la inicialización del proyecto. Inicializa archivo pyproject.toml, en el que se definen las dependencias del proyecto.

poetry add. Para adicionar paquetes, los cuales son incluidos como dependencias de forma autmática en el archivo pyproject.toml.

poetry update. Para actualizar el archivo poetry.lock con las ultimas versiones de dependencias instaladas.

poetry build. Para construir archivo .whl con el que se publicará el proyecto.

poetry install. Para resolver dependencias incluidas en achivo pyproject. toml e instalar el proyecto.

El resultado de configuración de dependencias se encuentra en el archivo <pyproject.toml>, al que se accede desde el link que se referencia más adelante.


Para el despliegue y ejecución del modelo se utilizó la libreria MLFlow, mediante los siguientes comandos básicos:

Inicialización del proyecto mediante la configuración del archivo MLproject

Creación de experimientos mediante mlflow experiments

Ejecución del proyecto mediante mlflow run

Generación del dashboard mediante mlflow server

Despliegue del modelo mediante mlflow models serve

mlflow models serve -m ""runs:/c8df84ac830d492685122538a67aa5bc/modft" --env-manager 

En el archivo  <MLproject> se especifica el nombre del proyecto, parámetros del modelo y ejecución del entrenamiento, el cual se encuentra en el link que se referencia más adelante.

  
  ## Lo anterior se puede revisar a profundidad en el enlace del código del paquete modft **https://github.com/margomeza16/mlds6_ft**. 

