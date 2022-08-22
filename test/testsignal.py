import numpy as np
import soundfile

samplerate = 48000

sig = np.zeros(10000)
sig[0] = 1
sig[-1] = 1

soundfile.write("impulse.wav", sig, samplerate)
