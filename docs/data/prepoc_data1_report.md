# Prepocesamiento dataset 1:

# train.zip.
Contiene el conjunto de 3200 imagenes de entrenamiento separadas en 4 clases de diagnóstico:["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"].
Cada clase contiene 800 imágenes. El procesamiento aplicado sobre este conjunto de imágenes es el siguiente:

1. Al cargar las imágenes en arreglo de numpy, mediante la función tf.keras.preprocessing.image.load_img, se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224*224, con representación RGB
2. Este resize se realiza debido a que se implementará un modelo de Fine Tunning para clasificar las imagenes, utilizando la red convolucional ResNet50V2 para la extracción de características generales,
3. la  cual requiere que el tamaño de las imágenes de entrada al modelo sean d 224*224 pixeles, con representación RGB.
