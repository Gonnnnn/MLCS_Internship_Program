#==========================================#
# Title:  Data Loader
# Author: Doohyun Lee
# Date:   2021-01-26
#==========================================#
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Input, Dropout, Concatenate, BatchNormalization, Activation, Bidirectional, GaussianNoise, Flatten
from keras.applications.densenet import DenseNet121
from keras.applications.resnet import ResNet50
from keras.regularizers import l2
from keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ImageClassification():
    def __init__(self):
        pass

    def build(self):
        input_tensor = tf.keras.Input(shape=(32,32,3)) # cifar10
        resized_tensor = tf.keras.layers.Lambda(lambda image : tf.image.resize(image, (224, 224)))(input_tensor)
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=resized_tensor, input_shape=(32,32,3), pooling='max', classes=1000)
        # base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=resized_tensor, input_shape=(224,224,3), pooling='max', classes=10)

        for layer in base_model.layers:
            layer.trainable = False

        output = base_model.output
        flatten_output = Flatten()(output)

        def fc(num_classes, _input, activation, trainable):
            x = _input
            x = Dense(512, kernel_regularizer=l2(0.001), trainable=trainable)(x)
            x = BatchNormalization(trainable=trainable)(x)
            x = Activation('relu', trainable=trainable)(x)
            x = Dropout(0.2)(x)

            x = Dense(512, kernel_regularizer=l2(0.001), trainable=trainable)(x)
            x = BatchNormalization(trainable=trainable)(x)
            x = Activation('relu', trainable=trainable)(x)
            x = Dropout(0.2)(x)
            return Dense(num_classes, activation=activation, trainable=trainable)(x)

        prediction = fc(10, flatten_output, 'softmax', True)

        model = Model(inputs=base_model.input, outputs=prediction)

        model.summary()

        model.compile(  loss="categorical_crossentropy",
                        optimizer='adam',
                        metrics=['accuracy'])

        return model