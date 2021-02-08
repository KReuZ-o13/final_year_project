# -*- coding: utf-8 -*-

from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import numpy as np
from scipy.io import wavfile

def silence_remover(filename):
    [Fs, x] = aIO.read_audio_file(filename)
    segments = aS.silence_removal(x, Fs, 0.020, 0.020, smooth_window = 0.1, weight = 0.6, plot = True)
    new_signal = np.concatenate(segments)
    wavfile.write('silence_removed.wav', Fs, new_signal)
    return new_signal

#file_name = input("Please enter the directory of the file.")
#Issues arise, why are x and y a single digit off?
file_name = 'D:/Program_Setups/pyAudioAnalysis/pyAudioAnalysis/data/3WORDS.wav'
silence_remover(file_name)