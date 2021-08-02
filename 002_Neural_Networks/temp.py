import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train)

print(x_train[2][0][0][0:3])