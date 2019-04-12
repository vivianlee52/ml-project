#! /usr/bin/python

import gzip
import numpy as np
import pandas as pd
from time import time
import IPython.display as display
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import seaborn as sns
import csv
sns.set()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getinfo(path):
    filenames = []
    labels = []
    if(path == "Mass_Train_Dataset"):
        with open("Mass_Train_Dataset/mass_case_train.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    i = i + 1
                    continue;
                elif i > 20:
                    break;
                else:
                    if(row[9] == "MALIGNANT"):
                        labels += [1]
                    else:
                        labels += [0]
                    filenames.append(str("Mass_Train_Dataset/Mass-Train/" + row[11].replace("\\", "/")))
                    i = i + 1
    else:
        with open("Mass-Testset/mass_case_test.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    i = i + 1
                    continue;
                elif i > 20:
                    break;
                else:
                    if(row[9] == "MALIGNANT"):
                        labels += [1]
                    else:
                        labels += [0]
                    filenames.append(str("Mass-Testset/Mass-Test/" + row[11].replace("\\", "/")))
                    i = i + 1
    return filenames, labels

def _parse_function(filename, label):
    img_raw = tf.read_file(filename)
    img_tensor = tf.image.decode_png(img_raw)
    image_resized = tf.image.resize_images(img_tensor, [32, 32])
    return image_resized, label

trainsetfilenames, trainsetlabels = getinfo("Mass_Train_Dataset")
print(trainsetfilenames, trainsetlabels)
traindsset = tf.data.Dataset.from_tensor_slices((trainsetfilenames, trainsetlabels))
trainset = traindsset.map(_parse_function)

testsetfilenames, testsetlabels = getinfo("Mass-Testset")
print(testsetfilenames, testsetlabels)
testdsset = tf.data.Dataset.from_tensor_slices((testsetfilenames, testsetlabels))
testset = traindsset.map(_parse_function)

model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

EPOCHS = 10
BATCH_SIZE = 128

steps_per_epoch = trainset.output_shapes.__getitem__(0).__len__()//BATCH_SIZE
validation_steps = testset.output_shapes.__getitem__(0).__len__()//BATCH_SIZE

train_generator = ImageDataGenerator().flow(trainset.output_shapes, trainsetlabels, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(testset.output_shapes, testsetlabels, batch_size=BATCH_SIZE)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_generator, validation_steps=validation_steps,
                    shuffle=True, callbacks=[tensorboard])

score = model.evaluate(testset.output_shapes.__getitem__(0), testset.output_shapes.__getitem__(1))
print('Test loss:', score[0])
print('Test accuracy:', score[1])
