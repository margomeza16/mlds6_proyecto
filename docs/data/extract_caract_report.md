# Reporte de extracción de características:

Teniendo en cuenta que se va a implementar un modelo de Fine Tunning, para la extracción de características se utilizará la red convolucional preeentrenada ResNet50V2, sin las capas densas del final,de la siguiente manera:

extractor = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False,
                                            input_shape=(224, 224, 3))
                                            
La extracción de características, consiste en la obtención de representaciones intermedias y las realiza de forma progresiva mediante las capas de convolución y de pooling conectadas de forma secuencial, hasta llegar a un nivel de abstracción suficiente, para que al adicionar capas densas de clasificación al final del modelo, se obtenga una predicción adecuada de la imágen. 

Para nuestro caso, una vez construido y entrenado el modelo se construyó función para la extracción del vector de característiscas de 4 imágenes aleatorias del conjunto de test, mediante la función predict del modelo, para determinar su clasificación. Asi mismo se construyo función para obtener la similitud del coseno entre dos vectores de características de dos imágenes distintas.

ResNet50V2. Es una versión modificada de la red inicial ganadora del primer lugar en el concurso de clasificaci´on ILSVRX 2015, esta cuenta con un total de 152
capas distribuidas con 50 bloques convolucionales, tiene aproximadamente 25.5 millones de parámetros dentro de su configuración. Esta utiliza una serie de de
bloques con redes residuales profundas (Deep residual Network), que mediante conexiones entre cada uno de los bloques convolucionales ayuda a mejorar la precisión
del modelo evitando incrementar el número de capas y parámetros. [Tomado del siguiente artículo: Kaiming He y col. “Deep residual learning for image recognition”. En: Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition. Vol. 2016-Decem. 2016, p´ags. 770-778. isbn: 9781467388504. doi:
10.1109/CVPR.2016.90.]
