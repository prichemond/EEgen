import numpy as np
import tensorflow as tf
from keras.layers import (Activation, AtrousConvolution1D, Convolution1D, Dense,
    Flatten, Input, Lambda, merge)
from keras.models import Model

NGPUS = 4

NFILTERS = 64
FILTERSIZE = 2
FILTERSTACK = 20

# Model definition.
# TODO : work in L2 regularization inside.

def wavenetBlock(atrous_n_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = AtrousConvolution1D(atrous_n_filters, atrous_filter_size,
                                       atrous_rate=atrous_rate,
                                       border_mode='same',
                                       activation='tanh')(input_)
        sigmoid_out = AtrousConvolution1D(atrous_n_filters, atrous_filter_size,
                                          atrous_rate=atrous_rate,
                                          border_mode='same',
                                          activation='sigmoid')(input_)
        merged = merge([tanh_out, sigmoid_out], mode='mul')
        skip_out = Convolution1D(1, 1, activation='relu', border_mode='same')(merged)
        out = merge([skip_out, residual], mode='sum')
        return out, skip_out
    return f

def get_basic_generative_model(input_size):
    input_ = Input(shape=(input_size, 1))
    A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2)(input_)
    skip_connections = [B]
    for i in range(FILTERSTACK):
        A, B = wavenetBlock(NFILTERS, FILTERSIZE, 2**((i+2)%9))(A)
        skip_connections.append(B)
    net = merge(skip_connections, mode='sum')
    net = Activation('relu')(net)
    net = Convolution1D(1, 1, activation='relu')(net)
    net = Convolution1D(1, 1)(net)
    net = Flatten()(net)
    net = Dense(256, activation='softmax')(net)
    model = Model(input=input_, output=net)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
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

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice,
                                     output_shape=input_shape,
                                     arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

def parallelize_and_compile(model):
    if NGPUS>1:
        model = make_parallel(model, NGPUS)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


simplemodel = get_basic_generative_model(256*64)
simplemodel.summary()
simplemodel = parallelize_and_compile(simplemodel)
