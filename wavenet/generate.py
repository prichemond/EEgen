import tensorflow as tf
import numpy as np
from keras.models import Model
import sys

frame_size = 256 * 64

#Generate signal (naively) from model, conditional on signal history
#TODO : implement fast generation. 

def generate_from_model(model, sr, duration, seed_signal):
    print('Generating signal...')
    new_signal = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_signal.shape[0]:
        softmax = np.array(model.predict(seed_signal.reshape(1,                                                               frame_size, 1)), dtype=float).reshape(256)
        softmax /= softmax.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=softmax)

        # Invert normalization & mu-law companding.
        ampl_val_8 = ((((predicted_val) / 255.0) - 0.5) * 2.0)
        ampl_val_16 = (np.sign(ampl_val_8) * (1/256.0) * ((1 + 256.0)**abs(
            ampl_val_8) - 1)) * 2**15
        new_signal[curr_sample_idx] = ampl_val_16
        seed_signal[-1] = ampl_val_16
        seed_signal[:-1] = seed_signal[1:]
        pc_str = str(round(100*curr_sample_idx/float(new_signal.shape[0]), 2))
        sys.stdout.write('Percent complete: ' + pc_str + '\r')
        sys.stdout.flush()
        curr_sample_idx += 1
    print('Signal generated.')
    return new_signal.astype(np.int16)


def fast_generate_from_model(model, sr, duration, seed_signal):
    #TODO : implement dyadic algorithm.
    return
