import numpy as np
import os
import tensorflow as tf
import pandas as pd
def preproceso(ruta_orig,batch_size):
	#El parámetro ruta_orig corresponde a la ruta donde se encuentran descargadas las imagenes a procesar. Debe terminar con el caracter "/"
	#
	all_images = []
	labels = []
	for i, val in enumerate(["0_normal/", "1_ulcerative_colitis/", "2_polyps/", "3_esophagitis/"]):
    	temp_path = f"{ruta_orig}{val}"
    #Se cargan las imagenes en un arreglo de numpy.
    # Se cambia el tamaño en pixeles de las imágenes de entrada (resizing), pasándolas a 224*224, con representación RGB
    for im_path in os.listdir(temp_path):
        all_images.append(np.array(tf.keras.preprocessing.image.load_img(temp_path+im_path,
                                                                         target_size=(224, 224, 3))))
        labels.append(i)
    if "train" in im_path:
    	X_train = np.array(all_images)
		y_train = np.array(labels)
	elif "test" in im_path:
		X_test = np.array(all_images)
		y_test = np.array(labels)
	elif "val" in im_path:
		X_val = np.array(all_images)
		y_val = np.array(labels)

	#Codificamos las etiquetas usando one-hot representation, una variable binaria por cada posible clase de salida:
	Y_train = tf.keras.utils.to_categorical(y_train)
	Y_test = tf.keras.utils.to_categorical(y_test)
	Y_val = tf.keras.utils.to_categorical(y_val)

	#Debido a que utilizaremos la red convlucional ResNet50V2 para la extracción de características,
	#utilizamos el preprocesamiento de la ResNet50V2 para transformar los conjuntos, escalando los pixeles de entrada entre -1 y 1:
  	X_train_prep = tf.keras.applications.resnet_v2.preprocess_input(X_train)
  	X_val_prep = tf.keras.applications.resnet_v2.preprocess_input(X_val)
  	X_test_prep = tf.keras.applications.resnet_v2.preprocess_input(X_test)

  	# Definimos las transformaciones para data augmentation
  	train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                fill_mode='constant')

  	#Se aplica data augmentation sobre el conjunto de imágenes de entrenamiento
  	train_gen = train_datagen.flow(X_train_prepro, Y_train, batch_size=batch_size)

  	return train_gen,X_train_prep,X_val_prep,X_test_prep