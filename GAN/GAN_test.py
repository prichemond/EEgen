import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime as time
import mne
import sklearn
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
print(raw_fname)

raw = mne.fiff.Raw(raw_fname)
print(raw)

start, stop = raw.time_to_index(100, 115)
data, times = raw[:, start:stop]
print(data.shape)
print(times.shape)
