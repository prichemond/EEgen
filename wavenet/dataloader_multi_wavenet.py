import numpy as np
import tensorflow as tf
import os
import sys

frame_size = 256 * 64

def get_signal(filename):
    sr, signal = read(filename)
    signal = signal.astype(float)
    # Normalize
    signal = signal - signal.min()
    signal = signal / (signal.max() - signal.min())
    signal = (signal - 0.5) * 2
    return sr, signal

#TODO : vectorize the below code

def frame_generator(sr, signal, frame_size, frame_shift, minibatch_size=20):
    signal_len = len(signal)
    X = []
    y = []
    while 1:
        for i in range(0, signal_len - frame_size - 1, frame_shift):
            frame = signal[i:i+frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= signal_len:
                break
            temp = signal[i + frame_size]
            
            # Mu-law 'companding' transform, quantizes on 8-bit.
            target_val = int((np.sign(temp) * (np.log(1 + 256*abs(temp)) / (
                np.log(1+256))) + 1)/2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []
