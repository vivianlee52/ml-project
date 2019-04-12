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

dir_path = os.path.dirname(os.path.realpath(__file__))

def getinfo(path):
    filenames = []
    labels = []
    if(path == "Mass_Train_Dataset"):
        with open("Mass_Train_Dataset/mass_case_train.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    pass
                elif i > 2:
                    break;
                else:
                    if(row[9] == "MALIGNANT"):
                        labels.append(1)
                    else:
                        labels.append(0)
                    filenames.append( dir_path + str("/Mass_Train_Dataset/Mass-Train/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    else:
        with open("Mass-Testset/mass_case_test.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    pass
                elif i > 2:
                    break;
                else:
                    if(row[9] == "MALIGNANT"):
                        labels.append(1)
                    else:
                        labels.append(0)
                    filenames.append( dir_path + str("/Mass-Testset/Mass-Test/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    print(filenames)
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    print(labels)
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [32,32])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    print(X, Y)
    return X, Y, labels

trainsetimages, trainsetlabels, r_trainsetlabels= getinfo("Mass_Train_Dataset")
print(trainsetimages, trainsetlabels)

testsetimages, testsetlabels, r_testsetlabels = getinfo("Mass-Testset")
print(testsetimages, testsetlabels)

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

print("finish compile")

batchsize = len(r_trainsetlabels)
test_size = len(r_testsetlabels)

trainsetlabels = to_categorical(r_trainsetlabels, 10)
testsetlabels = to_categorical(r_testsetlabels, 10)

print("after generator")

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(trainsetimages, trainsetlabels, steps_per_epoch=batchsize, verbose=2)

print("after fit")

score = model.evaluate(testsetimages, testsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
