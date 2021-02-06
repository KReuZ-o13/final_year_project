# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:19:35 2021

@author: KReuZ_o13
"""

from pydub import AudioSegment
import numpy as np
from scipy.io import wavfile
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly
import IPython

#The smarter solution than the speech-silence-classifier - no model, just math
fs, signal = wavfile.read("C:/Users/ADMIN1/Desktop/My Uni Work/FYP; ML for speech processing/Code/there_is_an_attempt_in_progress.wav")
signal = signal / (2**15)
signal_len = len(signal)
segment_size_t = 5 #segments are 5s long
segment_size = segment_size_t * fs  

segments = np.array([signal[x:x + segment_size] for x in
                     np.arange(0, signal_len, segment_size)])


'''for iS, s in enumerate(segments):
    wavfile.write("data/file_segment_{0:d}_{1:d}.wav".format(segment_size_t * iS,
                                                              segment_size_t * (iS + 1)), fs, (s))'''
    
energies = [(s**2).sum() / len(s) for s in segments]
thres = 0.4 * np.median(energies)
speech_segments_index = (np.where(energies > thres)[0])

segments2 = segments[speech_segments_index]


new_signal = np.concatenate(segments2)

wavfile.write("C:/Users/ADMIN1/Desktop/My Uni Work/FYP; ML for speech processing/Code/basic_audio_analysis/data/silence_free.wav", fs, new_signal)
plotly.offline.iplot({ "data": [go.Scatter(y=energies, name="energy"),
                                go.Scatter(y=np.ones(len(energies)) * thres, 
                                           name="thres")]})

