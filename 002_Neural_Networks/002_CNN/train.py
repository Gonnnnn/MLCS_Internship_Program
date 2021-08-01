from cnn_network import cifar_Res
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import random

batch_size = 128
epochs = 10
train_ratio = 0.7
val_ratio = 0.15

def cifar10_data_loader(train_ratio, val_ratio, shuffle = False):
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

    y_train, y_val, y_test = np.split(y_tot, [int(train_ratio * len(y_tot)), int((train_ratio + val_ratio) * len(y_tot))])
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, x_val, x_test, y_train, y_val, y_test

def main():
    model = cifar_Res().build()
    x_train, x_val, x_test, y_train, y_val, y_test = cifar10_data_loader(train_ratio, val_ratio, True)

    model.fit(x_train, y_train, epochs = epochs, verbose = 1, validation_data = (x_val, y_val))

    score = model.evaluate(x_test, y_test)
    print('Test loss of model:', score[0])
    print('Test accuracy of model:', score[1])

if __name__ == "__main__":
    main()