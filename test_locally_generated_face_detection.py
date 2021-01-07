from locally_generated_face_detection_model import creat_test_model
from keras.utils.io_utils import HDF5Matrix
import os
import numpy as np
import warnings
import imageio
from keras.utils import to_categorical
from scipy import misc
from sklearn import metrics
import cv2
import skimage
import random

model_weights = ""
test_model_d = creat_test_model(model_weights)

def transform_image(filename):
    image = misc.imread(filename)

    h, w, _ = image.shape
    if h != 256 or w != 256:
        out_image = misc.imresize(image, [256, 256], interp='bilinear')
    else:
        out_image = image

    return np.array(np.float32(out_image))

test_samples =np.load('./test_random'+'.npy')
num_batches = int(SAMPLES_test/64)
losses = 0
acc = 0
preciss = 0
recal = 0
num_b = 0
X_test = []
Y_test = []
f1_ave = 0
count= 0
for img in test_samples:
    a = img['image']
    images = np.array([transform_image(a)])
    annotations = np.array([np.int32(img['mask'], axis=-1)])
    X_test.append(images)
    Y_test.append(annotations)
    if len(X_test)==64:
        count = count+1
        num_b = num_b+1
        X_test = np.array(X_test)
        X_test = np.squeeze(X_test)
        Y_test = np.array(Y_test)
        pre_re = test_model_d.predict(X_test, verbose=1, batch_size=64)
        y_pre = pre_re[:,1]#pre_re.argmax(axis=1)#
        y_pre[y_pre<0.5]=0
        y_pre[y_pre>=0.5]=1
        accuracy = metrics.accuracy_score(Y_test, y_pre)
        precision = metrics.precision_score(Y_test, y_pre)
        recall = metrics.recall_score(Y_test, y_pre)
        print('acc:', accuracy, 'precision:', precision, 'recall:', recall)
        acc+=accuracy
        preciss +=precision
        recal += recall
        X_test = []
        Y_test = []

print('final batch')
num_b = num_b + 1
X_test = np.array(X_test)
X_test = np.squeeze(X_test)
Y_test = np.array(Y_test)
pre_re = test_model_d.predict(X_test, verbose=1, batch_size=SAMPLES_test-num_batches*64)
y_pre = pre_re[:, 1]
y_pre[y_pre < 0.5] = 0
y_pre[y_pre >= 0.5] = 1
accuracy = metrics.accuracy_score(Y_test, y_pre)
precision = metrics.precision_score(Y_test, y_pre)
recall = metrics.recall_score(Y_test, y_pre)
print('acc:', accuracy, 'precision:', precision, 'recall:', recall)
acc += accuracy
preciss += precision
recal += recall

print('acc:',acc/num_b,'precision:',preciss/num_b,'recall:',recal/num_b)
