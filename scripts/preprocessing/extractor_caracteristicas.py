import tensorflow as tf
def extraer_caracteristicas(data_train):
	extractor = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False,
                                            input_shape=(224, 224, 3))
	return extractor