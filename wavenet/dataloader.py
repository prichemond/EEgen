import numpy as np
import tensorflow as tf
import os
import time
import pickle
import h5py
from scipy.io.wavfile import read
from scipy.io import loadmat
from scipy.signal import resample, get_window

PROJECTOR_CHANNEL = 12

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
            # target_val = int((np.sign(temp) * (np.log(1 + 256 * abs(temp)) / (
            #    np.log(1 + 256))) + 1) / 2.0 * 255)
            target_val = int(255.0 * (1.0 + temp) / 2.0)
            # Here we apply no nonlinear transformation.

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
            return data
        except:
            print('Unable to load mat file ' + filename)
            return


def load_all_folder_matrices(folder, verbosity=False):
    starttime = time.time()
    all_filenames = tf.gfile.Glob(folder)
    number_filenames = len(all_filenames)
    if verbosity:
        input('Number of files to load : %s' % number_filenames)

    matlist = []
    for files in all_filenames:
        signal = load_filename_as_mat(files, verbosity)
        if not((np.count_nonzero(signal) < 10) or (np.any(np.std(signal, axis=0) < 0.5))):
            # Normalization step.
            signal -= np.min(signal)
            signal /= (np.max(signal) - np.min(signal))
            signal = (signal - 0.5) * 2.0

            matlist.append(signal)
        else:
            if verbosity:
                print('File is dropout, skipping...')
        print(len(matlist))

    if verbosity:
        print('Total time elapsed: %.3fs' % (time.time() - starttime))

    return matlist


def data_reduction(data):
    # As a temporary hack before learning 1x1 convolutions for BSS
    # Averaging spatially is pointless - for now we project.
    data = data[:, PROJECTOR_CHANNEL]
    # TODO : Try out Blackman-Harris or Chebyshev window resampling.
    data = resample(data, 600 * 256)

    return data


def load_dataset(frame_size, frame_shift):
    # Load full dataset into signal_train.
    print('Loading dataset...')
    file_train = open('./all_eeg_files.pkl', 'rb')
    train = pickle.load(file_train)
    # Turn the list of arrays into single array, via constructor & flattening.
    train = np.ravel(np.array(train))
    print('Dataset loaded.')
    # Split into training and validation.
    # Dirty for now, will call sklearn later
    split_fraction = 0.80
    split_number = int(split_fraction * float(train.shape[0]))
    val = train[split_number:-1]
    train = train[0:split_number]

    n_train_examples = (train.shape[0] - frame_size - 1
                        / float(frame_shift))
    print('Training examples ex validation : %s' % n_train_examples)
    return train, val


def pickled_file():
    matpathfilter = '/home/pierre/pythonscripts/AutoRegressive/eegnet/data/train/*.mat'
    alleegs = load_all_folder_matrices(matpathfilter, verbosity=True)
    # Serialize large file.
    with open('all_eeg_files.pkl', "wb") as pickle_file:
        pickle.dump(alleegs, pickle_file)

    return

# Unnecessary ever since the nd.array() constructor trick.


def aggregate_pickled_file():
    file_train = open('./all_eeg_files.pkl', 'rb')
    signal_train = pickle.load(file_train)
    sr = 256

    table = np.array(signal_train)
    print(table.shape)

    # Turn list of np_arrays into appended, 1d np_array before
    # getting companded frames.
    lensignal = len(signal_train)
    appendedsignal = signal_train[0]
    sizeone = 256 * 600
    for i in range(lensignal - 1):
        # This loop doesn't run in place and as such is slow.
        appendedsignal = np.append(appendedsignal, signal_train[i + 1])
        print(appendedsignal.shape[0] // sizeone)

    filesave_train = open('all_eeg_files_onego.pkl', 'wb')
    pickle.dump(appendedsignal, filesave_train)
    print('Aggregate file saved to disk.')
    return

# if __name__ == "main":
    pickled_file()
