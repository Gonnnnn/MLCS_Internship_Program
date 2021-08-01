# ==========================================#
# Title:  Data Loader
# Author: Doohyun Lee
# Date:   2021-01-26
# ==========================================#
from cnn_network import ImageClassification
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# parameter
batch_size = 256
epochs = 5
ratio1 = 0.7
ratio2 = 0.15


def split(dataset, label, ratio1, ratio2):
    [x_train, x_validation, x_test] = np.split(dataset, [int(len(dataset) * ratio1), int(len(dataset) * ratio2)])
    [y_train, y_validation, y_test] = np.split(label, [int(len(label) * ratio1), int(len(label) * ratio2)])

    # train_steps_per_epoch = x_train.shape[0] // batch_size
    # validation_steps_per_epoch = x_validation.shape[0] // batch_size

    # DenseNet
    x_train = tf.keras.applications.densenet.preprocess_input(x_train)
    y_train = to_categorical(y_train, 10)
    x_validation = tf.keras.applications.densenet.preprocess_input(x_validation)
    y_validation = to_categorical(y_validation, 10)
    x_test = tf.keras.applications.densenet.preprocess_input(x_test)
    y_test = to_categorical(y_test, 10)

    # ResNet
    # x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
    # y_train = to_categorical(y_train, 10)
    # x_validation = tf.keras.applications.resnet50.preprocess_input(x_validation)
    # y_validation = to_categorical(y_validation, 10)
    # x_test = tf.keras.applications.resnet50.preprocess_input(x_test)
    # y_test = to_categorical(y_test, 10)

    # train_datagen = ImageDataGenerator(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=20,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     horizontal_flip=True,
    #     rescale=1/255,
    #     zoom_range=0.2
    # )

    # test_datagen = ImageDataGenerator(
    #     rescale=1/255
    # )

    # train = train_datagen.flow(x=x_train, y=y_train, batch_size=batch_size)
    # validation = test_datagen.flow(x=x_validation, y=y_validation, batch_size=batch_size)
    # test = test_datagen.flow(x=x_test, y=y_test, batch_size=batch_size)

    # return train, validation, test, train_steps_per_epoch, validation_steps_per_epoch
    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def main():
    model = ImageClassification().build()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # train, validation, test, train_steps_per_epoch,validation_steps_per_epoch = split(x,y,ratio1,ratio2)
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = split(x, y, ratio1, ratio2)

    # model.fit(train,
    #         batch_size=batch_size,
    #         steps_per_epoch=train_steps_per_epoch,
    #         epochs=epochs,
    #         validation_data=validation,
    #         validation_steps=validation_steps_per_epoch,
    #         shuffle=True)

    model.fit(x_train, y_train,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validation, y_validation))

    score = model.evaluate(x_test, y_test)
    print('Test loss of model:', score[0])
    print('Test accuracy of model:', score[1])


if __name__ == "__main__":
    main()