#! /usr/bin/python

import gzip
import numpy as np
import pandas as pd
from time import time
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
    print(filename)
    image_string = tf.io.read_file(str(filename))
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [28, 28])
    return image_resized, label

filenames, labels = getinfo("Mass_Train_Dataset")
print(filenames)
print(labels)
trainsetfilenames = tf.constant(filenames)
trainsetlabels = tf.constant(labels)
trainset = tf.data.Dataset.from_tensor_slices((filenames, labels))
trainset = trainset.map(_parse_function)

filenames, labels = getinfo("Mass-Testset")
print(filenames)
print(labels)
testsetfilenames = tf.constant(filenames)
testsetlabels = tf.constant(labels)
testset = tf.data.Dataset.from_tensor_slices((filenames, labels))
testset = trainset.map(_parse_function)

print(trainset.output_shapes)

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

# X_train, y_train = train['features'], to_categorical(train['labels'])
# X_validation, y_validation = validation['features'], to_categorical(validation['labels'])
#
# train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
# validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)
#
# print('# of training images:', train['features'].shape[0])
# print('# of validation images:', validation['features'].shape[0])

steps_per_epoch = trainset.output_shapes.__getitem__(0).__len__()//BATCH_SIZE
validation_steps = testset.output_shapes.__getitem__(0).__len__()//BATCH_SIZE

print(trainset.output_shapes[0])
print(trainset.output_shapes[1])

train_generator = ImageDataGenerator().flow(trainset.output_shapes, trainsetlabels, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(testset.output_shapes, testsetlabels, batch_size=BATCH_SIZE)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                    validation_data=validation_generator, validation_steps=validation_steps,
                    shuffle=True, callbacks=[tensorboard])

score = model.evaluate(testset.output_shapes.__getitem__(0), testset.output_shapes.__getitem__(1))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""
def display_image(position):
    image = train['features'][position].squeeze()
    plt.title('Example %d. Label: %d' % (position, train['labels'][position]))
    plt.imshow(image, cmap=plt.cm.gray_r)

train = {}
test = {}

train['features'], train['labels'] = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test['features'], test['labels'] = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

print('# of training images:', train['features'].shape[0])
print('# of test images:', test['features'].shape[0])

train_labels_count = np.unique(train['labels'], return_counts=True)
dataframe_train_labels = pd.DataFrame({'Label':train_labels_count[0], 'Count':train_labels_count[1]})
dataframe_train_labels

validation = {}
train['features'], validation['features'], train['labels'], validation['labels'] = train_test_split(train['features'], train['labels'], test_size=0.2, random_state=0)

print('# of training images:', train['features'].shape[0])
print('# of validation images:', validation['features'].shape[0])

# Pad images with 0s
train['features']      = np.pad(train['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
validation['features'] = np.pad(validation['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')
test['features']       = np.pad(test['features'], ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(train['features'][0].shape))

//LeNet5
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
"""
