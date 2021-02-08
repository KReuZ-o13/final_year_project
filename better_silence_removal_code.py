# -*- coding: utf-8 -*-

from pydub import AudioSegment as aS
import numpy as np
from scipy.io import wavfile
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly
import IPython

def silence_remover (filename, file_type):
    #First, read and convert the signal
    if file_type == "wav" or "WAV" or "Wav":
        fs, signal = wavfile.read(filename)
    else:
        mp3_file = aS.from_mp3(filename)
        mp3_file.export("not_mp3.wav", format="wav")
        fs, signal = wavfile.read("not_mp3.wav")
    #normalise signal, and convert to mono, 
    signal = signal / (2**15)
    mono_sig = np.arange(0, len(signal)) / fs
    
    #Segment the signal
    sig_len = len(mono_sig)
    seg_t = 1
    seg_size = seg_t * fs 
    segments = np.array([signal[x:x + seg_size] for x in
                             np.arange(0, sig_len, seg_size)])
    
    #Find the energy of the signal to calculate its threshold and eliminate those below it
    energy = [(s**2).sum() / len(s) for s in segments]
    thres = 0.6 * np.median(energy)
    speech_seg_index = (np.where(energy > thres)[0])
    segments_2 = segments[speech_seg_index]
    new_signal = np.concatenate(segments_2)
    wavfile.write('silence_free.wav',fs, new_signal)
    plotly.offline.iplot({ "data": [go.Scatter(y=energy, name="energy"),
                                go.Scatter(y=np.ones(len(energy)) * thres, 
                                           name="thres")]}) 


audio_file = input("Please input the source path. \n")
audio_file_type =  input("Please give the file type. \n")
silence_remover(audio_file, audio_file_type) 
   
    
        
    