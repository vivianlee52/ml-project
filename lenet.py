import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend
from time import time
import seaborn as sns
import csv
from PIL import Image
from sklearn.metrics import roc_curve, auc
import os

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

def calc(testsetlabels, testresult):
    result = []
    for i in range(len(testsetlabels)):
        if(testsetlabels[i] == testresult[i]):
            result.append(1)
        else:
            result.append(0)
    return result

def getinfo(path, type):
    filenames = []
    labels = []
    if(path == "Calc_Train_Dataset"):
        with open("Calc_Train_Dataset/calc_case_train.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    pass
                else:
                    if(row[9] == "MALIGNANT"):
                        labels.append(1)
                    else:
                        labels.append(0)
                    if(type=='roi'):
                        filenames.append( dir_path + str("/Calc_Train_Dataset/Calc-Train/" + str(row[12]).replace("\\", "/")))
                    else:
                        filenames.append( dir_path + str("/Calc_Train_Dataset/Calc-Train/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    else:
        with open("Calc-Testset/calc_case_test.csv") as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in readCSV:
                if i == 0:
                    pass
                else:
                    if(row[9] == "MALIGNANT"):
                        labels.append(1)
                    else:
                        labels.append(0)
                    if(type=='roi'):
                        filenames.append( dir_path + str("/Calc-Testset/Calc-Test/" + str(row[12]).replace("\\", "/")))
                    else:
                        filenames.append( dir_path + str("/Calc-Testset/Calc-Test/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [256,256])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    return X, labels

type = 'full'
epochs = 30

trainsetimages, trainsetlabels = getinfo("Calc_Train_Dataset", type)

testsetimages, testsetlabels = getinfo("Calc-Testset", type)

model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(256,256,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(units=2, activation = 'softmax'))

model.summary()

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print("finish compile")

batchsize = len(trainsetlabels)
testsize = len(testsetlabels)

cattrainsetlabels = to_categorical(trainsetlabels, 2)
cattestsetlabels = to_categorical(testsetlabels, 2)

sess=tf.Session()
tf.train.start_queue_runners(sess)

print(trainsetimages)
print(testsetimages)

trainsetimgs = trainsetimages.eval(session=sess)
testsetimgs = testsetimages.eval(session=sess)

print(trainsetimgs[0])
print(trainsetimgs[1])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
"""
model.fit(trainsetimgs, cattrainsetlabels, epochs=epochs, verbose=1, shuffle=True)
print("finish fit")

model.save(type + str(epochs) + str(0.4) + ".h5")

score = model.evaluate(testsetimgs, cattestsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""

#ROC
model = load_model(type + str(epochs) + str(0.4) + ".h5")

y_pred = model.predict(testsetimgs)
y_pred = y_pred.ravel() #flatten both pred and true y
cattestsetlabels = cattestsetlabels.ravel()

print(y_pred)
print(cattestsetlabels)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(cattestsetlabels, y_pred)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='LeNet-5(area = {:.3f})'.format(auc_keras))
plt.legend(loc='best')
plt.title('ROC curve for LeNet')
plt.show()
plt.savefig( type + str(epochs) + '.png')

"""
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
"""
plt.show()
