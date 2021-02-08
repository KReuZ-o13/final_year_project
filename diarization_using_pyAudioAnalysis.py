# -*- coding: utf-8 -*-

from pyAudioAnalysis import audioSegmentation as aS


def diarize(filename):
    speech = aS.speaker_diarization(filename, 2)
    return speech

file_name = input('Please enter the directory of the file.')
diarize(file_name)