# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import IPython
import wave
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

#obtain input data
def input_read(filename):
    with wave.open(filename, 'r') as s:
        channels = s.getnchannels()
        assert channels == 1
        sample_width = s.getsampwidth()
        assert sample_width == 2
        rate = s.getframerate()
        assert rate == 16000
        frames = s.getnframes()
        audio_data = s.readframes(frames)
        duration = frames / rate
    return audio_data, rate, duration     

#Define an object called frames~
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

#So, deepspeech models need tiny chunks of data to work
#This function is used to create those tiny chunks
def segment_generator(frame_time, audio, rate):
    n = int(rate * (frame_time / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n
        
#Now let's use that function to get the frames!
def the_true_segment_generator (filename):
    audio, rate, audio_duration = input_read(filename)
    frames = segment_generator(30, audio, rate)
    segments = list(frames)
    return segments, rate, audio_duration


#file_name_1 = input("Path to the diarised wav files.")
file_name_1 = "D:/Project/deepspec/4507-16021-0012.wav"
segments, rate, audio_length = the_true_segment_generator(file_name_1)
f = open("transcript_1.txt", 'w+')
for i, segment in enumerate(segments):
    # Run deepspeech on the chunk
    audio = np.frombuffer(segment, dtype=np.int16)
    output = model.stt(audio)
    f.write(output[0] + " ")

# Summary of the files processed
f.close()