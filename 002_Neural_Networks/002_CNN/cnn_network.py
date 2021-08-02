import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.regularizers import l2

class cifar_Res():
    def __init__(self):
        pass

    def build(self):
        IMG_SHAPE = (32, 32, 3)
        input_tensor = tf.keras.Input(shape = IMG_SHAPE)
        resized = tf.keras.layers.Lambda(lambda image : tf.image.resize(image, (224, 224)))(input_tensor)
        res = ResNet50V2(input_shape=IMG_SHAPE, input_tensor=resized, include_top=False, pooling = 'max', weights='imagenet')

        for layer in res.layers:
            layer.trainable = False

        flatten = Flatten()(res.output)


        def fc(num_classes, _input, activation, trainable):
            a = _input
            a = Dense(512, kernel_regularizer = l2(0.001), trainable = trainable)(a)
            a = BatchNormalization(trainable = trainable)(a)
            a = Activation('relu', trainable = trainable)(a)
            a = Dropout(0.2)(a)

            a = Dense(256, kernel_regularizer=l2(0.001), trainable=trainable)(a)
            a = BatchNormalization(trainable=trainable)(a)
            a = Activation('relu', trainable=trainable)(a)
            a = Dropout(0.3)(a)

            return Dense(num_classes, activation = activation, trainable = trainable)(a)

        prediction = fc(10, flatten, 'softmax', True)
        model = tf.keras.models.Model(inputs = res.input, outputs = prediction)
        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        return model
