import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as layers
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
                #elif i > 2:
                #    break;
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
                #elif i > 2:
                #    break;
                else:
                    if(row[9] == "MALIGNANT"):
                        labels.append(1)
                    else:
                        labels.append(0)
                    filenames.append( dir_path + str("/Mass-Testset/Mass-Test/" + str(row[11]).replace("\\", "/")))
                i = i + 1
    print(filenames)
    print(labels)
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_images(image, [256,256])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    return X, labels

trainsetimages, trainsetlabels = getinfo("Mass_Train_Dataset")
print(trainsetimages, trainsetlabels)

testsetimages, testsetlabels = getinfo("Mass-Testset")
print(testsetimages, testsetlabels)

model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(256,256,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=2, activation = 'softmax'))

model.summary()

model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print("finish compile")

batchsize = len(trainsetlabels)
test_size = len(testsetlabels)

trainsetlabels = to_categorical(trainsetlabels, 2)
testsetlabels = to_categorical(testsetlabels, 2)

sess=tf.Session()
tf.train.start_queue_runners(sess)

trainsetimgs = trainsetimages.eval(session=sess)
testsetimgs = testsetimages.eval(session=sess)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit(trainsetimgs, trainsetlabels, steps_per_epoch=batchsize, epochs=1, verbose=1)
print("finish fit")

score = model.evaluate(testsetimgs, testsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_pred_keras = model.predict(testsetimgs).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(testsetlabels.ravel(), y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='LeNet-5 (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
plt.savefig('full.png')

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
