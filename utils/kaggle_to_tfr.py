"""
Created on Wed Feb  1 16:25:21 2017

@author: prichemond
"""
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
import time
import os
import shutil

# Read files list. Header: file, class (0: interictal, 1: preictal), safe (or not to use)
files_list = np.genfromtxt('./train_and_test_data_labels_safe.csv', 
                           dtype=("|S15", np.int32, np.int32), delimiter=',', skip_header=1)

# Get only files which are safe to use
files_list = [fl for fl in files_list if fl[2] == 1]

# Construct new file names based on class field
new_files_list = []
for fl in files_list:
    name = fl[0].split('.')[0].split('_')
    if len(name) == 3:
        name = name[0] + '_' + name[1] + '_' + str(fl[1]) + '.mat'
    else:
        name = name[0] + '_' + name[1] + 'test_' + str(fl[1]) + '.mat'
    new_files_list.append(name)

# Get only files names
files_list = [fl[0] for fl in files_list]

# Move files to new folder
print('Train data size:', len(files_list))
for idx in range(len(files_list)):
    print('Copying', files_list[idx], '----->', new_files_list[idx], 'index:', idx)
    shutil.copy('../data/train/'+files_list[idx], '../data/train_new/'+new_files_list[idx])
    
_SOURCE_FILES =  "../data/train/*.mat"
_DEST_FOLDER = "../dataset/train/"
_NUM_FILES = None # None is the total number of files

def mat2tfr(p_file, rem_dropout = False):
    # getting the filename and retrieving the patient, segement and label data
    pat, seg, label = p_file.split('/')[-1].split('.')[0].split("_")
    filename = pat + "_" + seg + "_" + label + ".tfr"
    fullpathname = _DEST_FOLDER + filename
    
    if os.path.exists(fullpathname):
        print("Dataset file", fullpathname, "already exists, skipping...")
    else: 
        t = time.time()    
        print("Converting " + p_file + " ----> " + fullpathname)
        # converting mat file as numpy
        mat = loadmat(p_file)
        data = mat['dataStruct']['data'][0][0]
        
        # Check if file is mainly zero's (100% dropout)
        if rem_dropout:
            if (np.count_nonzero(data) < 10) or (np.any(np.std(data, axis=0) < 0.5)):
                print("WARNING: File %s is all dropout." %p_file)
                return
             
        # TensorFlow Records writer
        with tf.python_io.TFRecordWriter(fullpathname) as tfrwriter:
            # Fill protobuff
            protobuf = tf.train.Example(features=tf.train.Features(feature={
                        'data' : tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten().tolist())), 
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])), 
                        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 
                    }))
            write = tfrwriter.write(protobuf.SerializeToString())
        elapsed = time.time() - t
        print("elapsed: %.3fs"%elapsed)
        
        
def dataset(folder, num_files=None):
    # get files
    filenames = tf.gfile.Glob(folder)
    # truncate reading
    if num_files is not None:
        filenames = filenames[:num_files]
    print("Converting #%d files."%len(filenames))
    
    for files in filenames:
        mat2tfr(files)       


dataset(_SOURCE_FILES, _NUM_FILES)

print('finished')