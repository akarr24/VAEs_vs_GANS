# -*- coding: utf-8 -*-


import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# keras import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


(X_train, Y_train),  (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
print(X_train.shape)
print(X_test.shape)
# Reshape the images
img_size = 28
X_train = X_train.reshape(-1, img_size, img_size, 1)
X_test = X_test.reshape(-1, img_size, img_size, 1)
print(X_train.shape)
print(X_test.shape)


model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(16, kernel_size=(5,5), activation=tf.nn.relu, input_shape=(28, 28, 1)),
                             tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                             tf.keras.layers.Dropout(rate=0.1),
                             tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu),
                             tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                             tf.keras.layers.Dropout(rate=0.1),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(64, activation=tf.nn.softmax),
                             tf.keras.layers.Dropout(rate=0.1),
                             tf.keras.layers.Dense(10),
                             tf.keras.layers.Activation(tf.nn.softmax)
                              
                                
                              
                              
                                
                      
])

model.compile(
   optimizer='adam',
   loss='categorical_crossentropy',  
   metrics=['accuracy'],
 )


history = model.fit(
    X_train,
    to_categorical(Y_train),
    epochs=25,  
    validation_data=(X_test, to_categorical(Y_test)), 
    batch_size=32
)