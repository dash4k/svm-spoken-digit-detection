# src/sound.py
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from settings import WAVE_OUTPUT_FILE, DURATION, SAMPLE_RATE

def record():
    print("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(WAVE_OUTPUT_FILE, SAMPLE_RATE, recording)
    print("Recording complete.")
