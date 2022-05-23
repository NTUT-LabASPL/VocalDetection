#!/bin/python
import librosa
import numpy as np
import os
import soundfile as sf
import argparse
import glob
import sys
import random
# PATH TO PROJECT ROOT
sys.path.append("/path/to/VocalDetection")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Split audio into multiple files and save analysis.')
parser.add_argument('-i', '--input', type=str, default='origin')
parser.add_argument('-o', '--output', type=str, default='output')
parser.add_argument('-s', '--sr', type=int, default=44100)
parser.add_argument('-c', '--count', type=int, default=0)
args = parser.parse_args()
count = args.count

filenames = []
analysis_folder = args.output

from models.SCNN18 import SCNN18

sample_size = 32000
input_shape = (sample_size, 1)
nb_classes = 2
model = SCNN18(input_shape, nb_classes).model()
model.load_weights('SCNN-Jamendo.h5')


def mkdir(folder):
    try:
        os.makedirs(folder)
    except Exception:
        pass


vocal_folder = os.path.join(analysis_folder, 'vocal')
non_vocal_folder = os.path.join(analysis_folder, 'non_vocal')
mkdir(vocal_folder)
mkdir(non_vocal_folder)


def basename(file):
    file = os.path.basename(file)
    return os.path.splitext(file)[0]


vocal_pairs = []
non_vocal_pairs = []
error_list = []

v_count = count
n_count = count


def split_wav(filename):
    global n_count
    global v_count
    y, sr = librosa.load(filename, sr=args.sr)

    samples_range = list(range(11, len(y) // args.sr - 11))

    fn = filename.replace('\\', '-').replace('.wav', '').replace(' ', '')
    vocal_count = 0
    non_vocal_count = 0
    while vocal_count != 2 or non_vocal_count != 2:
        if not samples_range:
            error_list.append(fn)
            break

        idx = np.random.randint(0, len(samples_range))
        val = samples_range.pop(idx)
        audio = y[args.sr * val:args.sr * (val + 2)]
        x = librosa.resample(audio, args.sr, 16000).reshape(1, 32000, 1)
        if model.predict(x)[0][0] > 0.5:
            if non_vocal_count == 2:
                continue
            non_vocal_count += 1
            n_count += 1
            filename = os.path.join(non_vocal_folder, f'n{n_count:04d}.wav')
            sf.write(filename, audio, sr)
            non_vocal_pairs.append(f'n{n_count:04d}.wav {fn}_{val}')

        else:
            if vocal_count == 2:
                continue
            vocal_count += 1
            v_count += 1
            filename = os.path.join(vocal_folder, f'v{v_count:04d}.wav')
            sf.write(filename, audio, sr)
            vocal_pairs.append(f'v{v_count:04d}.wav {fn}_{val}')


all_files = list(glob.iglob(f'./{args.input}/**/**.wav', recursive=True))
total = len(all_files)
random.shuffle(all_files)
i = 0
for filename in all_files:
    i += 1
    print(f'Processing file: {filename} , {i}/{total}')
    split_wav(filename)

np.savetxt(os.path.join(analysis_folder, 'meta_vocal.txt'), vocal_pairs, encoding='utf-8', fmt='%s')
np.savetxt(os.path.join(analysis_folder, 'meta_non_vocal.txt'), non_vocal_pairs, encoding='utf-8', fmt='%s')
np.savetxt(os.path.join(analysis_folder, 'error.txt'), error_list, encoding='utf-8', fmt='%s')
