#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 02:16:37 2017

@author: prichemond
"""
import mne as mne
from mne.datasets import sample

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

print(raw_fname)
raw = mne.fiff.Raw(raw_fname) 

print(raw)