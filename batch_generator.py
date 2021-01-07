import numpy as np
import random
from scipy import misc
from keras.utils import to_categorical
import os


def read_images(records, samples, batch_size, train=True):
    while True:
        if train:
            random.shuffle(records)
        for i in range(0, samples, batch_size):
            train_list = records[i:i+batch_size]
            # print(train_list)

            images = np.array([transform_image(filename['image']) for filename in train_list])
            annotations = np.array([np.int32(filename['mask'], axis=-1) for filename in train_list])
            annotations = to_categorical(annotations, num_classes=2)
            # print(images.shape)
            # print(annotations.shape)
            # Shuffle the data
            if train:
                perm = np.arange(images.shape[0])
                np.random.shuffle(perm)
                images = images[perm]
                annotations = annotations[perm]
            yield (images, annotations)


def transform_image(filename):
    image = misc.imread(filename)
    h, w, _ = image.shape
    if h != 256 or w != 256:
        out_image = misc.imresize(image, [256, 256], interp='bilinear')
    else:
        out_image = image
    
    return np.array(np.float32(out_image))
