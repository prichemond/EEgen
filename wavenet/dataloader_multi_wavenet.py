import numpy as np
import tensorflow as tf
import os
import time
from scipy.io.wavfile import read
from scipy.io import loadmat
from scipy.signal import resample, get_window


def load_signal(filename):
    sr, signal = read(filename)
    signal = signal.astype(float)
    # Normalize
    signal = signal - signal.min()
    signal = signal / (signal.max() - signal.min())
    signal = (signal - 0.5) * 2
    return signal, sr

# TODO : vectorize the below code.


def frame_generator(signal, sr, frame_size, frame_shift, minibatch_size=20):
    signal_len = len(signal)
    X = []
    y = []
    while 1:
        for i in range(0, signal_len - frame_size - 1, frame_shift):
            frame = signal[i:i + frame_size]
            if len(frame) < frame_size:
                break
            if i + frame_size >= signal_len:
                break
            temp = signal[i + frame_size]

            # Mu-law 'companding' transform, quantizes on 8-bit.
            target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (
                np.log(1 + 256))) + 1) / 2.0 * 255)
            X.append(frame.reshape(frame_size, 1))
            y.append((np.eye(256)[target_val]))
            if len(X) == minibatch_size:
                yield np.array(X), np.array(y)
                X = []
                y = []


def load_filename_as_mat(filename, verbose=False):
    if os.path.exists(filename):
        try:
            mat = loadmat(filename)
            data = mat['dataStruct']['data'][0][0]
            data = data_reduction(data)
            if verbose:
                print(filename)
                if (np.count_nonzero(data) < 10) or (np.any(np.std(data, axis=0) < 0.5)):
                    print('Warning : file %s is all dropout.' % filename)

        except:
            print('Unable to load mat file ' + filename)

    return data


def load_all_folder_matrices(folder, verbosity=False):
    starttime = time.time()
    all_filenames = tf.gfile.Glob(folder)
    number_filenames = len(all_filenames)
    if verbosity:
        input('Number of files to load : %f' % number_filenames)

    matlist = []
    for files in all_filenames:
        matlist.append(load_filename_as_mat(files, verbosity))

    if verbosity:
        print('Total time elapsed: %.3fs' % (time.time() - starttime))

    return matlist


def data_reduction(data):
    # TODO : Try out Blackman-Harris or Chebyshev window resampling.
    data = resample(data, 600 * 256)
    return data

matpathtrain = '/home/pierre/pythonscripts/AutoRegressive/eegnet/data/train/*.mat'
load_all_folder_matrices(matpathtrain, verbosity=True)
