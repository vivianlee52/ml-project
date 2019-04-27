import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.layers as layers
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend
import csv
import seaborn as sns
from numpy import genfromtxt
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

def getinfo(path, type, size, channel):
    filenames = []
    labels = []
    if(path == "Calc_Train_Dataset"):
        csvname = "/calc_case_train"
        subpath = "/Calc-Train/"
    else:
        csvname = "/calc_case_test"
        subpath = "/Calc-Test/"
    with open( path + csvname + ".csv") as csvfile:
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
                    filenames.append( dir_path + str("/" + path + subpath + str(row[12]).replace("\\", "/")))
                else:
                    filenames.append( dir_path + str("/" + path + subpath + str(row[11]).replace("\\", "/")))
            i = i + 1
    filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    filename, label = tf.train.slice_input_producer([filenames, labels], shuffle=True)
    image = tf.read_file(filename)
    image = tf.image.decode_png(image, channels=channel)
    image = tf.image.resize_images(image, [size,size])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    return X, labels

type = 'full'
epochs = 100

trainsetimages, trainsetlabels = getinfo("Calc_Train_Dataset", type, 32, 1)
testsetimages, testsetlabels = getinfo("Calc-Testset", type, 32, 1)


model = keras.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=2, activation = 'softmax'))
model.summary()
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

cattrainsetlabels = to_categorical(trainsetlabels, 2)
cattestsetlabels = to_categorical(testsetlabels, 2)
sess=tf.Session()
tf.train.start_queue_runners(sess)
print(trainsetimages)
print(testsetimages)
trainsetimgs = trainsetimages.eval(session=sess)
testsetimgs = testsetimages.eval(session=sess)
batchsize = len(trainsetlabels)
testsize = len(testsetlabels)

model.fit(trainsetimgs, cattrainsetlabels, epochs=epochs, verbose=1, shuffle=True)
model.save(type + str(epochs) + ".h5")

model = load_model(type + str(epochs) + ".h5")

score = model.evaluate(testsetimgs, cattestsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#ROC

y_pred = model.predict(testsetimgs)
np.savetxt('lenet.csv', y_pred, delimiter=",")
#y_pred = genfromtxt('lenet.csv', delimiter=',')
y_pred = y_pred.ravel() #flatten bot1h pred and true y
cattestsetlabels = cattestsetlabels.ravel()

fpr, tpr, thresholds = roc_curve(cattestsetlabels, y_pred)
auc = auc(fpr, tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='LeNet-5(area = {:.3f})'.format(auc))
plt.legend(loc='best')
plt.title('ROC curve')
plt.show()
plt.savefig('roc_lenet.png')
