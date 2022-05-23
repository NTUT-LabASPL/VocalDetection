import os

import librosa
import numpy as np
from math import sqrt

from definitions import DATA_DIR


class Noise:

    @classmethod
    def output_noise_audio(cls, file):
        samples, _ = librosa.load(file, sr=16000)
        x = np.sum(np.square(samples)) / len(samples)
        y = sqrt(x * 0.1)
        noise = np.random.normal(0, y, len(samples))
        samples = np.add(samples, noise)
        samples = np.clip(samples, -1, 1)
        return samples


class Pitch:

    @classmethod
    def output_pitch_audio(cls, file):
        samples, _ = librosa.load(file, sr=16000)
        y = librosa.effects.pitch_shift(samples, 16000, n_steps=-2)
        return y


# a = Noise().output_noise_audio(os.path.join(DATA_DIR, './jamendo/test/non_vocal/n_section0_0'))
# librosa.output.write_wav('./a.wav', a, 16000)
