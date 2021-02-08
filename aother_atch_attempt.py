# -*- coding: utf-8 -*-

import numpy as np
from pydub import AudioSegment as aS
from scipy.io import wavfile
import sys
import os
#import IPython
#import wave
import json
from deepspeech import Model

#define the model path and parameters. alpha, beta and beam width were the model training params
#tl;dr: they were lifted directly from the release page.
model_file_path = "D:/Project/deepspec/deepspeech-0.9.3-models.pbmm"
lm_file_path = "D:/Project/deepspec/deepspeech-0.9.3-models.scorer"
beam_width = 500
lm_alpha = 0.93
lm_beta = 1.18

model = Model(model_file_path)
model.enableExternalScorer(lm_file_path)

model.setScorerAlphaBeta(lm_alpha, lm_beta)
model.setBeamWidth(beam_width)
one_list = []

#obtain input data
def input_read(filename):
    rate, signal = wavfile.read(filename)
    return rate, signal 


#transcribe the audio and save it to one file
#considering  changing this from wave to pydub and using a for loop for the segments
#can I incorporate temp files?
#new issue: Where is this file getting the word entertainment in all the audio files I run?
#Or whatever that transcript is
#segments considered a tuple even though it was an array 
#Array conversion fail

def transcribe_this (audio_file):
    rate, signal = input_read(audio_file)
    signal = signal / (2**15)
    signal_len = len(signal)
    segment_size_t = 5 #segments are 1s long
    segment_size = segment_size_t * rate  
    
    segments = np.array([signal[x:x + segment_size] for x in
                     np.arange(0, signal_len, segment_size)])
    for i, segment in enumerate(segments):
        #wavfile.write(filename, rate, data)
        #asarray and array bring up same issue
        a = np.asanyarray(segment)
        data16 = np.frombuffer(a, dtype=np.int16)
        text_file = model.stt(data16)
        one_list.append(text_file)
    with open('transcript.txt','w') as f:
        f.write(json.dumps(one_list))
    print(text_file)
    return(text_file)
    
    
#lets use all the functions here, to obtain the file
#EOFE error. where is it popping up from....
#All input reads are bringing up an EOFE error
#Both wave and scipy
#Issue resolved, the file had been corrupted

input_file_1 = "D:/Project/test_audio.wav"
transcribe_this(input_file_1)