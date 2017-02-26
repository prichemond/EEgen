import numpy as np
import tensorflow as tf
from keras.layers import (Activation, AtrousConvolution1D, Convolution1D, Dense,
                          Flatten, Input, Lambda, merge)
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from keras.regularizers import l2
import time
import pickle
from dataloader import frame_generator

NFILTERS = 64
FILTERSIZE = 3
FILTERSTACK = 20

L2REGULARIZER = 0.00005
LEARNING_RATE = 0.001
NGPUS = 4


SGDOptimizer = SGD(lr=LEARNING_RATE, momentum=0.9, nesterov=True)
AdamOptimizer = Adam(lr=LEARNING_RATE)
NAdamOptimizer = Nadam(lr=LEARNING_RATE)
TRAIN_FLAG = False

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
        skip_out = Convolution1D(1, 1, activation='relu',
                                 border_mode='same',
                                 W_regularizer=l2(L2REGULARIZER))(merged)
        out = merge([skip_out, residual], mode='sum')
        return out, skip_out
    return f

# TODO : work in global L2 regularization within the skip-connections block.
# TODO : get average pooling in.


def get_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2)(input_)
    skip_connections = [B]
    for i in range(FILTERSTACK):
        A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2**((i + 2) % 9))(A)
        skip_connections.append(B)
    net = merge(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
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
    simplemodel = get_generative_model(256 * 64)
    simplemodel.summary()
    input('Parallelizing model, any key to continue.')
    simplemodel = parallelize_and_compile(simplemodel)
    simplemodel.summary()
    return simplemodel


def train_wavenet():

    wavenet = get_fullmodel()

    FRAME_SIZE = 256 * 64
    FRAME_SHIFT = 16
    N_EPOCHS = 1000
    S_EPOCHS = 3000

    file_train = open('./all_eeg_files.pkl', 'rb')
    signal_train = pickle.load(file_train)
    # Turn the list of arrays into single array, via constructor & flattening.
    signal_train = np.ravel(np.array(signal_train))
    sr = 256

    data_gen_train = frame_generator(
        signal_train, sr, FRAME_SIZE, FRAME_SHIFT)
    # Train statement
    if TRAIN_FLAG:
        wavenet.fit_generator(data_gen_train, samples_per_epoch=S_EPOCHS,
                              nb_epoch=N_EPOCHS, verbose=1)

        str_timestamp = str(int(time.time()))
        wavenet.save('models/model_' + str_timestamp +
                     '_' + str(N_EPOCHS) + '.h5')

    # except:
    #    print('Unable to load training data.')

    return

train_wavenet()