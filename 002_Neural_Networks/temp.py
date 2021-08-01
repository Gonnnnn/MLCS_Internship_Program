import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import RMSprop
import random
batch_size = 128
num_classes = 10
epochs = 20

def cifar10_data_loader(train_ratio, val_ratio, shuffle = False):

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_tot = np.concatenate((x_train, x_test), axis=0)
    # since /= is not allowed to change the data type of an instance, we have to change it w astype in advance
    x_tot = x_tot.astype('float32')
    y_tot = np.concatenate((y_train, y_test), axis=0)

    if shuffle == True:
        joined = list(zip(x_tot, y_tot))
        random.shuffle(joined)
        x_tot_temp, y_tot_temp = zip(*joined)
        x_tot = list(x_tot_temp)
        y_tot = list(y_tot_temp)

    x_train, x_val, x_test = np.split(x_tot, [int(train_ratio * len(x_tot)), int((train_ratio + val_ratio) * len(x_tot))])
    x_train /= 255
    x_val /= 255
    x_test /= 255

    #conver class vectors to binary class matrices
    y_train, y_val, y_test = np.split(y_tot, [int(train_ratio * len(y_tot)), int((train_ratio + val_ratio) * len(y_tot))])
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_val, x_test, y_train, y_val, y_test

x_train, x_val, x_test, y_train, y_val, y_test = cifar10_data_loader(0.7, 0.15, True)

def createRes(height, width, depth):
    IMG_SHAPE = (height, width, depth)
    res = ResNet50V2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    for layer in res:
        layer.trainable = False
    res.summary()

    classifier = res.output
    classifier = GlobalAveragePooling2D()(classifier)
    classifier = Dense(128, activation='relu')(classifier)
    classifier = Dense(128, activation='relu')(classifier)
    classifier = Dense(num_classes, activation='softmax')(classifier)

    model = tf.keras.Model(inputs = res.input, outputs = classifier)
    model.summary()

    learning_rate = 0.0001
    model.compile(loss='categorical_crossentropy',
                   optimizer=RMSprop(lr=learning_rate),
                   metrics=['accuracy'])

    return model

def evaluate(model, batch_size, epochs, x_test, y_test):

    history = model.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(x_val, y_val))

    result = model.evaluate(x_test, y_test)
    print('Test loss1:', result[0])
    print('Test accuracy1:', result[1])

