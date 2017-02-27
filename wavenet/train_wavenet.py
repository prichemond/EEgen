import numpy as np
import tensorflow as tf
from keras.layers import (Activation, AtrousConvolution1D, Convolution1D, Dense,
                          Flatten, Dropout, Input, Lambda, merge)
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
import time
import pickle
from dataloader import load_dataset, frame_generator

NFILTERS = 64
FILTERSIZE = 3
# Looks like currently we are limited to 5 stacking blocks if 9 filtermod
# Due to a bug in 'atrous' not running more than a 280 ( 256 really ) rate on Tensorflow.
# Stack Overflow issue : https://github.com/fchollet/keras/issues/5529
FILTERSTACK = 20
FILTERMOD = 7

L2REGULARIZER = 0.00005
LEARNING_RATE = 0.001
NGPUS = 4

FRAME_SIZE = 256 * 32
FRAME_SHIFT = 32
N_EPOCHS = 2000
S_EPOCHS = 3000


SGDOptimizer = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
AdamOptimizer = Adam(lr=LEARNING_RATE)
NAdamOptimizer = Nadam(lr=LEARNING_RATE)


# Model definition.
def wavenetBlock(atrous_n_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = AtrousConvolution1D(atrous_n_filters, atrous_filter_size,
                                       atrous_rate=atrous_rate,
                                       border_mode='same',
                                       W_regularizer=l2(L2REGULARIZER),
                                       activation='tanh')(input_)
        sigmoid_out = AtrousConvolution1D(atrous_n_filters, atrous_filter_size,
                                          atrous_rate=atrous_rate,
                                          border_mode='same',
                                          W_regularizer=l2(L2REGULARIZER),
                                          activation='sigmoid')(input_)
        merged = merge([tanh_out, sigmoid_out], mode='mul')
        # Could add batchnorm here like so. Way slow though :
        merged = BatchNormalization()(merged)
        skip_out = Convolution1D(1, 1, border_mode='same',
                                 W_regularizer=l2(L2REGULARIZER),
                                 activation='relu')(merged)
        out = merge([skip_out, residual], mode='sum')
        return out, skip_out
    return f

# TODO : work in global L2 regularization within the skip-connections block.
# TODO : get average pooling or dropout in.


def get_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2)(input_)
    skip_connections = [B]
    for i in range(FILTERSTACK):
        A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2**((i + 2) % FILTERMOD))(A)
        skip_connections.append(B)
    net = merge(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    # Some extra regularization in the fully-connected layer :
    net = Dropout(0.5)(net)
    model = Model(input=input_, output=net)

    return model


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)

        size = tf.concat([shape[:1] // parts, shape[1:]], 0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], 0)

        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


def parallelize_and_compile(model):
    if NGPUS > 1:
        model = make_parallel(model, NGPUS)

    model.compile(loss='categorical_crossentropy', optimizer=NAdamOptimizer,
                  metrics=['accuracy'])
    return model


def get_fullmodel():
    simplemodel = get_generative_model(FRAME_SIZE)
    simplemodel.summary()
    print('Parallelizing model')
    simplemodel = parallelize_and_compile(simplemodel)
    simplemodel.summary()
    return simplemodel


def train_wavenet():
    # Build model.
    wavenet = get_fullmodel()
    input('Press any key.')

    signal_train, signal_val = load_dataset(FRAME_SIZE, FRAME_SHIFT)
    sr = 256
    # Start training via 'frame_generator' online one-hot & quantizing function
    data_gen_train = frame_generator(signal_train, sr, FRAME_SIZE, FRAME_SHIFT)
    data_gen_val = frame_generator(signal_val, sr, FRAME_SIZE, FRAME_SHIFT)

    # Add relevant training helpers ('callbacks' in Keras parlance.)
    csv_cb = CSVLogger('./logs/trainlog.csv')
    modelsave_cb = ModelCheckpoint('./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                   verbose=1)
    tensorboard_cb = TensorBoard(log_dir='./logs', histogram_freq=5)
    # Train statement
    # For now : no callbacks ; check nb_val_samples.
    wavenet.fit_generator(data_gen_train, samples_per_epoch=S_EPOCHS, nb_epoch=N_EPOCHS,
                          validation_data=data_gen_val, nb_val_samples=FRAME_SIZE,
                          verbose=1, callbacks=[modelsave_cb, csv_cb, tensorboard_cb])

    str_timestamp = str(int(time.time()))
    wavenet.save('models/model_' + str_timestamp +
                 '_' + str(N_EPOCHS) + '.hdf5')

    return

train_wavenet()
