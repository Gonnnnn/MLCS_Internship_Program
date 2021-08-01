import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization


class cifar_Res():
    def __init__(self):
        pass

    def build(self):
        IMG_SHAPE = (32, 32, 3)
        input_tensor = tf.keras.Input(shape = (224, 224))
        resized = tf.keras.layers.Lambda(lambda image : tf.image.resize(image, (224, 224)))(input_tensor)
        res = ResNet50V2(input_shape=IMG_SHAPE, include_top=False, pooling = 'max', weights='imagenet', )

        for layer in res:
            layer.trainable = False

        flatten = Flatten()(res.output)


        def fc(num_classes, _input, activation, trainable):
            l2 = tf.keras.regularizers.l2()
            a = _input
            a = Dense(512, kernel_regularizer = l2(0.001), trainable = trainable)(a)
            a = BatchNormalization(trainable = trainable)(a)
            a = Activation('relu', trainable = trainable)(a)
            a = Dropout(0.2)(a)

            a = Dense(256, kernel_regularizer=l2(0.001), trainable=trainable)(a)
            a = BatchNormalization(trainable=trainable)(a)
            a = Activation('relu', trainable=trainable)(a)
            a = Dropout(0.3)(a)

            return Dense(num_classes, activation = activation, trainable = trainable)

        prediction = fc(10, flatten, 'softmax', True)
        model = tf.keras.models.Model(inputs = res.input, outputs = prediction)
        model.summary()
        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        return model
