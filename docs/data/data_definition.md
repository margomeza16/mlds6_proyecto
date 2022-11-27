# Data and Feature Definitions

This document provides a central hub for the raw data sources, the processed/transformed data, and feature sets. More details of each dataset is provided in the data summary report. 

For each data, an individual report describing the data schema, the meaning of each data field, and other information that is helpful for understanding the data is provided. If the dataset is the output of processing/transforming/feature engineering existing data set(s), the names of the input data sets, and the links to scripts that are used to conduct the operation are also provided. 

For each dataset, the links to the sample datasets in the _**Data**_ directory are also provided. 

_**For ease of modifying this report, placeholder links are included in this page, for example a link to dataset 1, but they are just placeholders pointing to a non-existent page. These should be modified to point to the actual location.**_

## Raw Data Sources

| Dataset Name | Original Location   | Destination Location  | Data Movement Tools / Scripts | Link to Report |
| ---:| ---: | ---: | ---: | -----: |
|train |Los datos para entrenamiento se descargan manualmente,  como archivo zip, con la opción "Download" de la siguiente página de Kaggle, ingresando con usuario registrado,  https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning. Posteriormente se suben manualmente a la siguiente dirección de GoogleDrive: https://drive.google.com/file/d/1BwSoPrJzTLndqMjBMThOWr-qQe7FbOyU/view?usp=share_link, para de allí ser descargados en ruta temporal de GoogleColab.  | El archivo train.zip se descarga y descomprime en la siguiente ruta temporal de GoogleColab: /tmp/train/  | [descarga_file.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/data_acquisition/descarga_file.py) | [Dataset 1 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/data_dictionary.md)|
| test| Los datos para test se descargan manualmente,  como archivo zip, con la opción "Download" de la siguiente página de Kaggle, ingresando con usuario registrado,  https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning. Posteriormente se suben manualmente a la siguiente dirección de GoogleDrive: https://drive.google.com/file/d/1AQWlyBOm9EG6tQOSk5HNbpHkpSCSVr0Y/view?usp=share_link, para de allí ser cargados en ruta temporal de GoogleColab.| El archivo test.zip se descarga y descomprime en la siguiente ruta temporal de GoogleColab: /tmp/test/ | [descarga_file.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/data_acquisition/descarga_file.py) | [Dataset 2 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/data_dictionary.md)|
| val| Los datos para validación se descargan manualmente,  como archivo zip, con la opción "Download" de la siguiente página de Kaggle, ingresando con usuario registrado, https://www.kaggle.com/datasets/francismon/curated-colon-dataset-for-deep-learning. Posteriormente se suben manualmente a la siguiente dirección de    GoogleDrive: https://drive.google.com/file/d/1TBS_el84d3lEgrNTbXezzLt6T1SYk0mV/view?usp=share_link, para de allí ser descargados en ruta temporal de GoogleColab.| El archivo val.zip se descarga y descomprime en la siguiente ruta temporal de GoogleColab: /tmp/val/ | [descarga_file.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/data_acquisition/descarga_file.py) | [Dataset 3 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/data_dictionary.md)|

* Dataset1 summary. <Provide brief summary of the data, such as how to access the data. More detailed information should be in the Dataset1 Report.>

# train.zip:
Archivo zip descargado de la página de Kaggle, desde la dirección indicada en el cuadro anterior. Contiene 3200 imágenes de endoscopías en formato JPG, con diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles, que corresponden al conjunto de entrenamiento, separadas en 4 carpetas según el diagnóstico del especialista, cada clase de diagnóstico contiene 800 imágenes. El archivo se encuentra en GoogleDrive y se puede acceder  a través del siguiente link: https://drive.google.com/file/d/1BwSoPrJzTLndqMjBMThOWr-qQe7FbOyU/view?usp=share_link.

* Dataset2 summary. <Provide brief summary of the data, such as how to access the data. More detailed information should be in the Dataset2 Report.> 

# test.zip:
Archivo zip descargado de la página de Kaggle, desde la dirección indicada en el cuadro anterior. Contiene 800 imágenes de endoscopías en formato JPG, con diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles, que corresponden al conjunto de pruebas, separadas en 4 carpetas según la clase de diagnóstico del especialista, cada clase de diagnóstico contiene 200 imágenes. El archivo se encuentra en GoogleDrive y se puede acceder  a través del siguiente link: https://drive.google.com/file/d/1AQWlyBOm9EG6tQOSk5HNbpHkpSCSVr0Y/view?usp=share_link.

* Dataset3 summary. <Provide brief summary of the data, such as how to access the data. More detailed information should be in the Dataset3 Report.> 

# val.zip:
Archivo zip descargado de la página de Kaggle, desde la dirección indicada en el cuadro anterior. Contiene 2000 imágenes de endoscopías en formato JPG, con diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles, que corresponden al conjunto de pruebas, separadas en 4 carpetas según la clase de diagnóstico del especialista, cada clase de diagnóstico contiene 500 imágenes. El archivo se encuentra en GoogleDrive y se puede acceder  a través del siguiente link: https://drive.google.com/file/d/1TBS_el84d3lEgrNTbXezzLt6T1SYk0mV/view?usp=share_link.

## Processed Data
| Processed Dataset Name | Input Dataset(s)   | Data Processing Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| train.zip | [Dataset1](https://drive.google.com/file/d/1BwSoPrJzTLndqMjBMThOWr-qQe7FbOyU/view?usp=share_link)| [preproc_img.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/preprocessing/prepoc_img.py) | [Processed Dataset 1 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/preproc_report.md)|
| test.zip | [Dataset2]([link/to/dataset2/report](https://drive.google.com/file/d/1AQWlyBOm9EG6tQOSk5HNbpHkpSCSVr0Y/view?usp=share_link)) |[preproc_img.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/preprocessing/prepoc_img.py) | [Processed Dataset 2 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/preproc_report.md)|
| val.zip | [Dataset3]([link/to/dataset2/report](https://drive.google.com/file/d/1TBS_el84d3lEgrNTbXezzLt6T1SYk0mV/view?usp=share_link)) |[preproc_img.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/preprocessing/prepoc_img.py) | [Processed Dataset 3 Report](https://github.com/margomeza16/mlds6_proyecto/blob/master/docs/data/preproc_report.md)|
* Processed Data1 summary. <Provide brief summary of the processed data, such as why you want to process data in this way. More detailed information about the processed data should be in the Processed Data1 Report.>

# train.zip:

Al conjunto de imagenes de entrenamiento se aplicaron los siguientes preprocesamientos:

Se cargan en arreglo de numpy y mediante la función tf.keras.preprocessing.image.load_img, se les cambia el tamaño a 224*224 pixeles con representación RGB. Esto se hace debido a que se utilizará la red convolucional preentrenada ResNet50V2 para la extracción de características, la cual requiere que las imágenes de entrada tengan esa resolución.

A este arreglo de imágenes de numpy, se aplica el procesamiento de la ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, que es requerido para escalar los pixeles de las imágenes entre -1 y 1.

Finalmente y debido a que la red convolucional requiere gran cantidad de imágenes, sobre el procesamiento anterior se aplica data augmentation, para obtener más imagenes de entrenamiento transformando el conjunto original (cambios en traslación, rotación, intensidad, entre otros).

Por último, a las etiquetas de las 4 clases de diagnóstico: ["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"], se les aplica códificación one-hot, para tener una representación binaria por cada clase de salida.


* Processed Data2 summary. <Provide brief summary of the processed data, such as why you want to process data in this way. More detailed information about the processed data should be in the Processed Data2 Report.> 

# test.zip:
Contiene el conjunto de imágenes de prueba, sobre las cuales se aplican los siguientes preprocesamientos:

Se cargan en arreglo de numpy y mediante la función tf.keras.preprocessing.image.load_img, se les cambia el tamaño a 224*224 pixeles con representación RGB. Esto se hace debido a que se utilizará la red convolucional preentrenada ResNet50V2 para la extracción de características, la cual requiere que las imágenes de entrada tengan esa resolución.

A este arreglo de imágenes de numpy, se aplica el procesamiento de la ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, que es requerido para escalar los pixeles de las imágenes entre -1 y 1.

Por último, a las etiquetas de las 4 clases de diagnóstico: ["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"], se les aplica códificación one-hot, para tener una representación binaria por cada clase de salida.

* * Processed Data3 summary. <Provide brief summary of the processed data, such as why you want to process data in this way. More detailed information about the processed data should be in the Processed Data3 Report.> 

# val.zip:
Contiene el conjunto de imágenes de validación, sobre las cuales se aplican los siguientes preprocesamientos:

Se cargan en arreglo de numpy y mediante la función tf.keras.preprocessing.image.load_img, se les cambia el tamaño a 224*224 pixeles con representación RGB. Esto se hace debido a que se utilizará la red convolucional preentrenada ResNet50V2 para la extracción de características, la cual requiere que las imágenes de entrada tengan esa resolución.

A este arreglo de imágenes de numpy, se aplica el procesamiento de la red convolucional ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input, que es requerido para escalar los pixeles de las imágenes entre -1 y 1.

Por último, a las etiquetas de las 4 clases de diagnóstico: ["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"], se les aplica códificación one-hot, para tener una representación binaria por cada clase de salida.

## Feature Sets

| Feature Set Name | Input Dataset(s)   | Feature Engineering Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| train.zip | [Dataset1](https://drive.google.com/file/d/1BwSoPrJzTLndqMjBMThOWr-qQe7FbOyU/view?usp=share_link) | [extractor_caracteristicas.py](https://github.com/margomeza16/mlds6_proyecto/blob/master/scripts/preprocessing/extractor_caracteristicas.py) | [Feature Set1 Report](link/to/report1)|


* Feature Set1 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set1 Report.>

Teniendo en cuenta que se va a implementar un modelo de Fine Tunnig, para la extracción de características del conjunto de imágenes de entrenamiento, se utiliza la red convolucional preeentrenada ResNet50V2, sin las capas densas del final.
* Feature Set2 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set2 Report.> 
