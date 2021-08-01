import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import random

batch_size = 128
num_classes = 10
epochs = 20

def mnist_data_loader(train_ratio, val_ratio, shuffle = False):

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #concat and split into train, val and test
    x_tot = np.concatenate((x_train, x_test), axis=0)
    x_tot = x_tot.reshape(70000, 784)
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


x_train, x_val, x_test, y_train, y_val, y_test = mnist_data_loader(0.7, 0.15, True)

#model 1
def model1():
    model1 = Sequential()
    model1.add(Dense(512, activation='relu', input_shape=(784,)))
    model1.add(Dropout(0.2))
    model1.add(Dense(512, activation='relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(num_classes, activation='softmax'))
    model1.summary()
    model1.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model1.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val))
    return model1

#model2
def model2():
    model2 = Sequential()
    model2.add(Dense(512, activation='relu', input_shape=(784,)))
    model2.add(Dropout(0.2))
    model2.add(Dense(256, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(128, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(num_classes, activation='softmax'))
    model2.summary()
    model2.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    history = model2.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_val, y_val))
    return model2
