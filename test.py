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
filename = dir_path + '/Mass_Train_Dataset/Mass-Train/P_00001/LEFT_CC_1/full.png'
image_file = tf.read_file(filename)
image_decoded = tf.image.decode_png(image_file, channels=1)
with tf.Session() as sess:
     f, img = sess.run([image_file, image_decoded])
     print(f[:20])
     print(img[:20])

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
                        labels += [1]
                    else:
                        labels += [0]
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
                        labels += [1]
                    else:
                        labels += [0]
                    filenames.append( dir_path + str("/Mass-Testset/Mass-Test/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [32,32])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    return X, Y

trainsetimages, trainsetlabels = getinfo("Mass_Train_Dataset")
print(trainsetimages, trainsetlabels)

testsetimages, testsetlabels = getinfo("Mass-Testset")
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

EPOCHS = 10
BATCH_SIZE = 128

steps_per_epoch = tf.size(trainsetlabels)//BATCH_SIZE
validation_steps = tf.size(testsetlabels)//BATCH_SIZE

train_generator = ImageDataGenerator().flow(trainsetimages, to_categorical(tf.Session().run(trainsetlabels)), batch_size=steps_per_epoch)
validation_generator = ImageDataGenerator().flow(testsetimages, to_categorical(tf.Session().run(testsetlabels)), batch_size=validation_steps)

print("after generator")

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=1,
                    validation_data=validation_generator, validation_steps=validation_steps,
                    shuffle=True, callbacks=[tensorboard])

print("after fit_generator")

score = model.evaluate(testsetimages, testsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
