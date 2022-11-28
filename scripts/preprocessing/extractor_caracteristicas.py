import tensorflow as tf
import numpy as np
import pandas as pd
import os, random
def get_modelResNet50V2(train_gen,X_train,X_val_prep,Y_val,batch_size):
	extractor = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False,
                                            input_shape=(224, 224, 3))
	
	# congelamos el extractor de características y adicionamos capas para clasificación: 
	for layer in extractor.layers:
    	    layer.trainable=False

	#Capa de global average pooling
	pool = tf.keras.layers.GlobalAveragePooling2D()(extractor.output)
	#capa densa con 32 neuronas y activación relu
	dense1 = tf.keras.layers.Dense(32, activation="relu")(pool)
	#Capa de dropout con taza de 0.2 para regularización
	drop1 = tf.keras.layers.Dropout(0.2)(dense1)
	#Capa densa de salida con 15 clases
	dense2 = tf.keras.layers.Dense(4, activation="softmax")(drop1)
	# definimos nuestro modelo de fine tunning
	ft_model = tf.keras.models.Model(inputs=[extractor.input], outputs=[dense2])

	# compilamos el modelo
	ft_model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-3),
    	             metrics=["accuracy"])

	# Entrenamos el modelo por 2 épocas para el calentamiento:
	ft_model.fit_generator(train_gen, validation_data=(X_val_prepro, Y_val),
    	                   epochs=2, steps_per_epoch=X_train.shape[0]//batch_size)
	
	#Descongelamos las ultimas 13 capas de la red convolucional para entrenarlas junto con las 4 capas densas de clasificación adicionadas al modelo
	for layer in ft_model.layers[-17:]:
    	layer.trainable = True

    # disminuímos el learning rate y compilamos el modelo
	ft_model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=1e-4),
                 	metrics=["accuracy"])

	ft_model.summary()



	# definimos el callback
	best_callback = tf.keras.callbacks.ModelCheckpoint(filepath="fine_tuning.h5", monitor="val_loss", 
    	                                               verbose=True, save_best_only=True,
    	                                               save_weights_only=True, mode="min")

	# entrenamos el modelo
	hist_ft = ft_model.fit(train_gen, validation_data=(X_val_prep, Y_val),
    	                             epochs=20, steps_per_epoch=X_train.shape[0]//batch_size,
    	                             callbacks=[best_callback])
	
	return ft_model, hist_ft, best_callback


#Función para obtener el vector de características de una imagen aleatoria de cada clase del conjunto de test

def getFeatureVector(model, img_path):
	class_dict = {0:'0_normal', 1:'1_ulcerative_colitis', 2:'2_polyps', 3:'3_esophagitis'}
	images = []
	for i in range(4):
		cat_path = f"/tmp/test/{class_dict[i]}/"
  		img = tf.keras.preprocessing.image.load_img(cat_path+random.choice(os.listdir(cat_path)),
        	                                        target_size=(224, 224, 3))
  		images.append(np.array(img))
	
	all_images = np.array(images)

	imgs_prep = tf.keras.applications.resnet_v2.preprocess_input(all_images)
	fecture_vectors = tl_model.predict(imgs_prep)
	return fecture_vectors


# Get cosine similarity between feature vectors A and B using cosine similarity
def getCosineSimilarity(A, B):
  cos_similarity = np.dot(A,B.T) / (np.linalg.norm(A)*np.linalg.norm(B)) # Get cosine similarity
  return cos_similarity[0][0]
