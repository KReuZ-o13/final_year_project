# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 07:39:06 2021

@author: KReuZ_o13
"""

import numpy as np
import sys
import os
import IPython
import wave
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
 

#obtain input data
def input_read(filename):
    with wave.open(filename, 'rb') as s:
        rate = s.getframerate()
        frames = s.getnframes()
        buffer = s.readframes(frames)
    return buffer, rate 


#transcribe the audio and save it to one file
#considering  changing this from wave to pydub and using a for loop for the segments
#can I incorporate temp files?
def transcribe_this (audio_file):
    buffer, rate = input_read(audio_file)
    data16 = np.frombuffer(buffer, dtype=np.int16)
    text_file = model.stt(data16)
    print(text_file)
    return(text_file)
    
    
#lets use all the functions here, to obtain the file
#EOFE error. where is it popping up from....
#All input reads are bringing up an EOFE error
#Both wave and scipy
input_file_1 = "D:/Project/deepspec/4507-16021-0012.wav"
transcribe_this(input_file_1)