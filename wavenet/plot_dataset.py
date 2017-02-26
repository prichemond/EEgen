import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from scipy.io import loadmat
from scipy.signal import resample
import pandas as pd
import time

NUM_FIGS = 100


# Generate a multi-channel plot from a raw single .mat file.
def plot_single_matfile(filename, verbose=False):
    starttime = time.time()
    try:
        if os.path.exists(filename):
            mat = loadmat(filename)
            data = mat['dataStruct']['data'][0][0]
            # Resample and plot in figure.
            data = resample(data, 600 * 256)
            dataframe = pd.DataFrame(data)
            # Looks like a bug in the 'colormap' option.
            # Either the 'colormap' option or the below line won't seem to work.
            # plt.set_cmap('hot')
            dataframe.plot(subplots=True, figsize=(48, 36))

            filename_end = filename.split('/')
            filename_end = filename_end[-1]
            plt.savefig('./figures/figure_' +
                        filename_end.split('.')[0] + '.png')

            if verbose:
                print('Plot saved. Time elapsed: %.2fs' %
                      (time.time() - starttime))

            return
        """
        plt.figure(figsize=(48, 36))
        plt.title('Training matrix : ' + filename_end)
        plt.tight_layout()
        sharedxaxis = np.arange(600 * 256)
        figs, axes = plt.subplots(nrows=16, ncols=1, sharex=True)
        for i in range(16):
            print(data[:, i])
            axes[i].plot(data[:, i])

                """
    except:
        print('File exists, but unable to load mat file ' + filename)
        return


# Generate NUM_FIGS plots from pickled training data, randomly picked.
def plot_dataset_subset(filename):
    # Matplotlib style from the book 'Bayesian methods for Hackers'
    plt.style.use('bmh')
    try:
        f = open(filename, 'rb')
        signal = pickle.load(f)
        signum = len(signal)
        permutation = np.random.choice(np.arange(signum), size=NUM_FIGS)

        for i in range(NUM_FIGS):
            index = permutation[i]
            smallsig = signal[index]
            # Now onto the plotting & saving.
            plt.figure(figsize=(48, 18))
            plt.title('Training data number : ' + str(index))
            plt.plot(smallsig)
            plt.savefig('./figures/figure_' + str(index) + '.png')
            # Adding basic stats.
            print(index)
            print(np.min(smallsig))
            print(np.max(smallsig))
            print(smallsig[0:1024])

    except:
        print('Unable to open file path, aborting.')

    return

plot_dataset_subset('./all_eeg_files.pkl')
train_path = '/home/pierre/pythonscripts/AutoRegressive/eegnet/data/train/'
#plot_single_matfile(train_path + '1_531_0.mat', verbose=True)
