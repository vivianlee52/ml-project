import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import load_model
import keras.layers as layers
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
import seaborn as sns
import csv
from PIL import Image
from sklearn.metrics import roc_curve, auc
import os

from keras.applications.vgg16 import VGG16

sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dir_path = os.path.dirname(os.path.realpath(__file__))

def getinfo(path, type):
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
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize_images(image, [224,224])
    X, Y = tf.train.batch([image, label], batch_size=tf.size(filenames))
    return X, labels

type = 'full'
epochs = 10

trainsetimages, trainsetlabels = getinfo("Calc_Train_Dataset", type)
testsetimages, testsetlabels = getinfo("Calc-Testset", type)
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

model = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=2)

model.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

model.fit(trainsetimgs, cattrainsetlabels, epochs=epochs, verbose=1, shuffle=True)
model.save(type + str(epochs) + "vgg.h5")
print("Finish fit")

score = model.evaluate(testsetimgs, cattestsetlabels)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#ROC
model = load_model(type + str(epochs) + "vgg.h5")
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
plt.title('ROC curve for LeNet-5')
plt.show()
plt.savefig( type + str(epochs) + 'vgg.png')
