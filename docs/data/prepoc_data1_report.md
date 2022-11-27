# Prepocesamiento:

# train.zip.
Contiene un conjunto de 3200 imagenes de entrenamiento separadas en 4 carpetas que corresponden a las clases de diagnóstico:["0: normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"].
Cada clase contiene 800 imágenes. 

Sobre este conjunto de imágenes, que se encuentran en formato JPG, con diferentes resoluciones, que van desde 720x576 hasta 1920x1072 píxeles, se aplica el siguiente preprocesamiento:

1. Resizing. Al cargar las imágenes en arreglo de numpy, mediante la función tf.keras.preprocessing.image.load_img, se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224*224 pixeles, con representación RGB. Este resize se realiza debido a que se implementará un modelo de Fine Tunning para clasificar las imagenes, utilizando la red convolucional ResNet50V2 para la extracción de características generales, la  cual requiere que el tamaño de las imágenes de entrada al modelo sean d 224*224 pixeles, con representación RGB. Este arreglo de numpy se almacena en la variable X_train, mientras que las etiquetas de clasifición de cada una de las imágenes, según la clase del diagnóstico al que pertenezcan, se almacenan en la variable y_train,  asignándole alguno de los siguientes valores numéricos: 0 para diagnóstico normal, 1 para colitis ulcerativa, 2 para polipos y 3 para esofaguitis. 

2. tf.keras.applications.resnet_v2.preprocess_input(X_train). Al conjunto X_train resultante del paso anterior se aplica el preprocesamiento de la red convolucional ResNet50V2 para escalar entre -1 y 1 los pixeles de las imágenes de entrada entrada que tienen un tamaño de 224*224 pixeles. Este preprocesamiento se debe aplicar a todas las imágenes que entren al modelo. La salida de este preprocesamiento se almacena en la variable X_train_prep

3. Data augmentation. Debido a que las redes convolucionales requieren una gran cantidad de imágenes para que pueden aprender, al conjunto de entrenamiento resultante del paso anterior se aplica data augmentation, para generar nuevas imágenes transformadas (cambios en traslación, rotación, intensidad, entre otros) a partir del conjunto de imágenes original. De esta forma se amplia el tamaño del conjunto de imágenes de entrenamiento. La siguiente es la definición de las transformaciones realizadas al conjunto de entrenamiento:

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='constant')

4. One-hot. Sobre la variable y_train, que contiene las etiquetas de clasificación del diagnóstico de cada imagen de entrenamiento, se aplica codificación one-hot para obtener una representación binaria de cada clase de diagnóstico de salida. El siguiente es el código aplicado: Y_train = tf.keras.utils.to_categorical(y_train)
