import tensorflow as tf
import numpy as np
from keras.models import Model
import sys


def get_signal_from_model(model, sr, duration, frame_size, seed_signal):
    print('Generating signal...')
    new_signal = np.zeros((sr * duration))
    curr_sample_idx = 0
    while curr_sample_idx < new_signal.shape[0]:
        distribution = np.array(model.predict(seed_signal.reshape(1,                                                               frame_size, 1)), dtype=float).reshape(256)
        distribution /= distribution.sum().astype(float)
        predicted_val = np.random.choice(range(256), p=distribution)
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
