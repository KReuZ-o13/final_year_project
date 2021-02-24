# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:38:45 2021

@author: KReuZ_o13
"""

import numpy as np
import sys
import os
import IPython
import wave
import json
import webrtcvad
import collections
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
#Check the audio parameters (rate at 16000, channels at mono)
#Gets the audio duration and the data
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
#Used to store the instances we'll create from the frames
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
        
'''#Now let's use that function to get the frames!
def the_true_segment_generator (filename):
    audio, rate, audio_duration = input_read(filename)
    frames = segment_generator(30, audio, rate)
    segments = list(frames)
    return segments, rate, audio_duration'''

#This removes any silence we didn't remove, and pads the frames generated

def vad_collects_all(rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

#The true segment generator, lol

def the_true_segment_generator(filename, aggressiveness):
    audio, rate, audio_length = input_read(filename)
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = segment_generator(30, audio, rate)
    frames = list(frames)
    segments = vad_collects_all(rate, 30, 300, vad, frames)

    return segments, rate, audio_length

#file_name_1 = input("Path to the diarised wav files.")
file_name_1 = "D:/Project/deepspec/4507-16021-0012.wav"
segments, rate, audio_length = the_true_segment_generator(file_name_1, 1)
f = open("transcript_1.txt", 'w+')
for i, segment in enumerate(segments):
    # Run deepspeech on the chunk
    audio = np.frombuffer(segment, dtype=np.int16)
    output = model.stt(audio)
    f.write(output[0] + " ")

# Summary of the files processed
f.close()