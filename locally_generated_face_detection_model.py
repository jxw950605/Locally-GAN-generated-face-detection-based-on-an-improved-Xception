from keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Layer,
    Input,
    Activation,
    BatchNormalization,
)
from keras.layers.core import *
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import backend as K
from batch_generator import read_images
from keras.layers import *
from keras.models import Model as KerasModel
from pickle_data import read_dataset
import numpy as np
import keras
import tensorflow as tf
import warnings
import os
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
warnings.filterwarnings("ignore")

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "", "batch size for training")
tf.flags.DEFINE_integer("NUM_EPOCHS_TRAIN", "", "NUM_EPOCHS for training")
tf.flags.DEFINE_string(
    "weights_path",
    "",
    "path to save model weights",
)
tf.flags.DEFINE_string("model_weights", "", "saved weights")
tf.flags.DEFINE_float("learning_rate", "", "Learning rate for Optimizer")

######################

reduction_ratio=4
def squeeze_excitation_layer(x, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=out_dim // reduction_ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, out_dim))(excitation)

    scale = multiply([x, excitation])

    return scale
#######################


######################

class Preprocess(Layer):
    def call(self, x, mask=None):
        # substract channel means if necessary
        # xout = preprocess_input(x, mode="tf")
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, 3)

def fake_locally(img_shape=(256, 256, 3)):
    def InceptionLayer(a, b, c, d, name1='Inception'):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate=2, strides=1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate=3, strides=1, padding='same', activation='relu')(x4)

            y = Concatenate(axis=-1)([x1, x2, x3, x4])

            return y

        return func

    img_input = Input(shape=img_shape)
    # Block 1
    x = InceptionLayer(6, 10,10, 6, name1='x1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = squeeze_excitation_layer(x, 32)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(x)

    x = InceptionLayer(12, 20, 20, 12, name1='x2')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = squeeze_excitation_layer(x, 64)


    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    # Block 2
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_layer(x, 128)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('elu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_layer(x, 256)
    C3 = add([x, residual])

    residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False)(C3)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('elu')(C3)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_layer(x, 512)
    x = add([x, residual])
    C4 = x
    # Block 5 - 8
    for i in range(4):
        residual = C4

        x = Activation('elu')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = squeeze_excitation_layer(x,512)
        x = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation('elu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_layer(x, 728)
    x = add([x, residual])


    # Block 14
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    C5 = x
    feature_size = 256
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced1')(C5)
    P4 = Add(name='addP51')([
        UpSampling2D((2,2),name='upP51')(P5),
        Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced1')(C4)])
    P3 = Add(name='addP41')([
        UpSampling2D((2, 2), name='upP41')(P4),
        Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced1')(C3)])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='p3_cn1')(P3)
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='p4_cn1')(P4)
    P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='p5_cn1')(P5)
    # Fully Connected Layer


    x_fc = GlobalAveragePooling2D(name = 'avp31')(P3)
    x_fc = Dropout(0.5)(x_fc)
    x_fc = Dense(16,name='fc31')(x_fc)

    x_fc4 = GlobalAveragePooling2D(name = 'avp41')(P4)
    x_fc4 = Dropout(0.5)(x_fc4)
    x_fc4 = Dense(16,name='fc41')(x_fc4)


    x_fc5 = GlobalAveragePooling2D(name = 'avp51')(P5)
    x_fc5 = Dropout(0.5)(x_fc5)
    x_fc5 = Dense(16,name='fc51')(x_fc5)

    xc = Concatenate()([x_fc,x_fc4])
    xc = Concatenate()([xc, x_fc5])
    x_fc = LeakyReLU(alpha=0.1)(xc)
    x_fc = Dropout(0.5)(x_fc)
    x_fc = Dense(2, activation='softmax',name='ac1')(x_fc)
    inputs = img_input
    # Create model
    model = Model(inputs, x_fc, name='xception_improved')
    return model


def main(argv=None):
    # prepare dataset
    training_samples =np.load('./train_random'+'.npy')
    validation_samples =np.load('./validation_random'+'.npy')
    SAMPLES_TRAIN = len(training_samples)
    SAMPLES_VALID = len(validation_samples)
    NUM_TRAIN_STEPS = int(np.ceil(SAMPLES_TRAIN / FLAGS.batch_size))
    NUM_VALID_STEPS = int(np.ceil(SAMPLES_VALID / FLAGS.batch_size))
    print("NO. %d of training samples" % SAMPLES_TRAIN)
    print("NO. %d of validation samples" % SAMPLES_VALID)
    # define model
    faces_detection = fake_locally()
    train_model = Model(
        inputs=faces_detection.inputs,
        outputs=faces_detection.layers[-1].output,
        name="fake_faces_detection_Featex",
    )
    def Recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        possible_positives = K.sum(y_true)
        possible_positives = K.cast(possible_positives, "float32")
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def Precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        y_true = K.flatten(K.argmax(y_true))
        y_pred = K.flatten(K.argmax(y_pred))
        true_positives = K.sum(y_true * y_pred)
        true_positives = K.cast(true_positives, "float32")
        predicted_positives = K.clip(
            K.cast(K.sum(y_pred), "float32"), K.epsilon(), None)
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    opt = keras.optimizers.Adam(lr=FLAGS.learning_rate, decay=0.000001)
    train_model.compile(optimizer=opt, loss=["categorical_crossentropy"], metrics=["acc",Recall,Precision])

    train_model.summary()

    # define save_model callback
    if not os.path.exists(FLAGS.weights_path + "/model"):
        os.makedirs(FLAGS.weights_path + "/model")

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(
            FLAGS.weights_path, "weights_{epoch:02d}-{val_acc:.4f}.hdf5"
        ),
        save_weights_only=True,
        verbose=1,
        save_best_only=False,
        mode="min",
    )
    path = os.path.join(FLAGS.weights_path, FLAGS.model_weights)
    if os.path.exists(path):
        train_model.load_weights(path, by_name=True)
        print("INFO: successfully restore train_weights from: {}".format(path))
    else:
        print("INFO: train_weights: {} not exist".format(path))

    train_model.fit_generator(
        generator=read_images(training_samples, SAMPLES_TRAIN, FLAGS.batch_size),
        steps_per_epoch=NUM_TRAIN_STEPS,
        epochs=FLAGS.NUM_EPOCHS_TRAIN,
        validation_data=read_images(
            validation_samples, SAMPLES_VALID, FLAGS.batch_size, train=False
        ),
        validation_steps=NUM_VALID_STEPS,
        callbacks=[checkpointer],
        workers=5,
        use_multiprocessing =True
    )

    # save Model
    train_model.save_weights(os.path.join(FLAGS.weights_path, "train_all_weights.h5"))


def creat_test_model(model=None):
    # define model
    faces_detection = fake_locally()
    train_model = Model(
        inputs=faces_detection.inputs,
        outputs=faces_detection.layers[-1].output,
        name="fake_faces_detection_locally",
    )

    if model is not None:
        try:
            train_model.load_weights(model)
            print("INFO: successfully load pretrained weights from {}".format(model))
        except Exception as e:
            print(
                "INFO: fail to load pretrained weights from {} for reason: {}".format(
                    model, e
                )
            )
    return train_model

if __name__ == "__main__":
    tf.app.run()
