import numpy as np
import matplotlib.pyplot as plt
from mne.io import read_raw_edf
from mne.datasets import eegbci

# #############################################################################
# # Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]

# strip channel names of "." characters

for raw in raws:
    raw.rename_channels(lambda x: x.strip('.'))
    raw.filter(0.5, None)

plt.close('all')
raws[0].plot(scalings=dict(eeg=100e-6))
raws[0].plot_psd(tmax=np.inf, average=True)
