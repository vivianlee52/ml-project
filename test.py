#! /usr/bin/python

import gzip
import numpy as np
import pandas as pd
from time import time
#from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
import csv
sns.set()

def read_x():
    trainlabels = []
    trainfeatures = []
    testlabels = []
    testfeatures = []
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
                file = open("Mass_Train_Dataset/Mass-Train/"+ row[12].replace("\\", "/"), 'rb' )
                print(row[12])
                trainlabels.append(np.frombuffer(file.read()))
                i = i + 1
        print(trainlabels)
        trainlabels = np.array(trainlabels, dtype=np.uint8)
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
                file = open("Mass_Train_Dataset/Mass-Train/"+ row[11].replace("\\", "/"), 'rb' )
                trainfeatures += np.frombuffer(file.read(), dtype=np.uint8)
                i = i + 1
        trainfeatures = np.array(trainfeatures, dtype=np.uint8)
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
                file = open("Mass-Testset/Mass-Test/"+ row[12].replace("\\", "/"), 'rb' )
                testlabels += np.frombuffer(file.read(), dtype=np.uint8)
                i = i + 1
        testlabels = np.array(testlabels, dtype=np.uint8)
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
                file = open("Mass-Testset/Mass-Test/"+ row[11].replace("\\", "/", 'rb') )
                testfeatures += np.frombuffer(file.read(), dtype=np.uint8)
                i = i + 1
        testfeatures = np.array(testfeatures, dtype=np.uint8)
    return trainfeatures, trainlabels, testfeatures, testlabels
train = {}
test = {}
train['features'], train['labels'], test['features'], test['labels'] = read_x()

print('# of training images:', train['features'].shape[0])
print('# of test images:', test['features'].shape[0])

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
