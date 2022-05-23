import sys
# PATH TO PROJECT ROOT
sys.path.append("/path/to/VocalDetection")
import scipy
from utils.noise import Noise, Pitch
from collections import OrderedDict
import shutil
from definitions import *
import json
import librosa
import fnmatch
import os
from datetime import datetime
from random import shuffle, random
import random
import numpy as np
import h5py
import math
import zipfile
from numpy import array
from time import sleep
import re
from definitions import DATA_DIR, LOG_DIR, WEIGHT_DIR, DATASET_DIR, ROOT_DIR

# order Dict


def find_files(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


class JamendoAudioReader:

    def __init__(self, use_mfcc=False, has_bootstrap=False, bootstrap_num=0, output_num=0):
        self.output_num = output_num

        # retrieve_tags_and_audios & split_audios_by_tags's parameter
        self.data = os.path.join(ROOT_DIR, './data')
        self.training_data = os.path.join(self.data, './jamendo2s/')
        self.audio_file_names = find_files(self.training_data)

        # retrieve_to_hadoop's parameter
        self.test_music = os.path.join(self.data, './jamendo2s/')
        self.out_data = os.path.join(self.data, './dataset/')
        self.output_vocal_dir = os.path.join(self.out_data, './vocal')
        self.output_non_vocal_dir = os.path.join(self.out_data, './non_vocal')

        self.retrieve_dataset = [
            # os.path.join(self.data, './InHouseTestData'),
            # os.path.join(self.data, './FMA-C-1-fixed-classification/test'),
            # os.path.join(self.data, './FMA-C-2-fixed-classification/train'),
            # os.path.join(self.data, './FMA-C-1-fixed-classification/test/right'),
            # os.path.join(self.data, './FMA-C-1-fixed-classification/train'),
            # os.path.join(self.data, './FMA-C-1-fixed-classification/uscl-jamendo-predict-0'),
            # os.path.join(self.data, './ktv-mpg-dataset/audio2s/test')
            # os.path.join(self.data, './Jamendo-Stereo-USCL/train')
            # '/media/dl/Data/all_dataset/30%/'
            # '/media/dl/Data/all_dataset/new30%/',
            # '/home/dl/DeepLearningVocalClassification/del/',
            # '/home/mino/DeepLearningVocalClassification/data/jamendo2s/test/',
            # '/home/mino/DeepLearningVocalClassification/data/KTV-Autoencoder/test/',
            # os.path.join(DATA_DIR, 'FMA-C-1-fixed-classification', 'train')
            # '/Data/TrainingMusic/台語2s/stream/train'
            '/Data/TrainingMusic/古典歌劇純音樂2s/test'
            # os.path.join(DATA_DIR, 'KTV', 'test')
            # '/media/dl/Data/A-Cappella/'
            # '/media/dl/Data/all_dataset/new30%/',
            # '/home/dl/DeepLearningVocalClassification/data/Jamendo-Stereo-USCL/train/left_copy/',
        ]
        self.retrieve_dir = ['./vocal', './non_vocal']
        # self.retrieve_dir = ['./non_vocal']
        # self.retrieve_dir = ['./vocal_ae_minus', './non_vocal_ae_minus']

        self.output_left_vocal_dir = os.path.join(self.out_data, './left/vocal')
        self.output_left_non_vocal_dir = os.path.join(self.out_data, './left/non_vocal')
        self.output_right_vocal_dir = os.path.join(self.out_data, './right/vocal')
        self.output_right_non_vocal_dir = os.path.join(self.out_data, './right/non_vocal')

        self.output_merge_dir = os.path.join(self.out_data, './merged')
        self.output_uncertain_dir = os.path.join(self.out_data, './uncertain')

        # boostrap
        self.has_bootstrap = has_bootstrap
        if has_bootstrap:
            self.boostrap_music_dir = os.path.join(self.data, './fusionBootstrap/uscl_convlstm_mfcc_mfcc/50persent/{}'.format(bootstrap_num))
            self.boostrap_vocal_dir = os.path.join(self.boostrap_music_dir, './vocal')
            self.boostrap_non_vocal_dir = os.path.join(self.boostrap_music_dir, './non_vocal')
            self.output_hadoop_file = os.path.join(self.data, './191106-fmac2-test-{}.h5'.format(bootstrap_num))

        # os.path.join(self.data, './180713-Jamendo-Test-mfcc-40bands-noShuffle-{}.h5'.format(output_num))
        self.output_hadoop_file = os.path.join(DATASET_DIR, './SCNN-Classical-test.h5')
        # self.output_hadoop_file = os.path.join(self.data, './FMA-C-1-SCNN-Stereo-Mono-Version-Test.h5')
        self.output_2s_wav = False
        self.shuffle_all_files = True
        self.tagging_mapping_audio_name = OrderedDict()
        self.sample_rate = 44100
        # self.out_sample_rate = 15552
        # self.sample_size = 31104
        self.out_sample_rate = 16000
        self.sample_size = 32000
        self.stereo_mode = False
        self.stereo_two_file = False
        self.binary_mode = False
        self.fft_mode = False
        # 必須確保兩個資料夾下的檔案為平行且命名格式有順序 shuffle_all_files 必須設定為False
        self.autoencoder_mode = False
        self.hop_size = 0.5
        self.use_mfcc = use_mfcc
        self.mfcc_hop_size = 512
        self.n_fft = 2048
        self.n_mfcc = 80
        self.mfcc_size = math.ceil(self.sample_size / float(self.out_sample_rate) * 31.4 * self.out_sample_rate / 16000 * 512 / self.mfcc_hop_size)
        self.fft_row = 1025
        self.fft_col = 63
        self.constantQ_mode = False
        self.ceptrom_mode = False
        self.Hamming_widowsize = 1024
        self.Hamming = np.hamming(self.Hamming_widowsize)
        self.cep_stride = 512
        self.cepsize = self.sample_size // self.cep_stride

        self.iirt_mode = False
        self.turning = 0.8

    def set_input_data_dir(self, directory):
        self.training_data = os.path.join(self.data, directory)
        self.audio_file_names = find_files(self.training_data)
        self.tagging_mapping_audio_name = OrderedDict()

    def set_out_dir(self, directory):
        self.out_data = os.path.join(self.data, directory)
        self.output_vocal_dir = os.path.join(self.out_data, './vocal')
        self.output_non_vocal_dir = os.path.join(self.out_data, './non_vocal')

    def retrieve_tags_and_audios(self):
        tags_file = os.path.join(self.data, 'jamendo_lab.zip')

        self.tag_zip = zipfile.ZipFile(tags_file)
        tag_file_names = self.tag_zip.namelist()

        for name in tag_file_names:
            matches = re.search('\/(.*).lab', name)
            if matches is None:
                continue
            file_name = matches.group(1)
            matching_audio_file_name = [audio_file_name for audio_file_name in self.audio_file_names if file_name in audio_file_name]
            if len(matching_audio_file_name) > 0:
                self.tagging_mapping_audio_name[name] = matching_audio_file_name[0]

        print("======================")
        for name in self.tagging_mapping_audio_name:
            print(name)

    def split_audios_by_tags(self):
        if self.stereo_mode:
            self.split_stereo_audios_by_tags()
        elif self.stereo_two_file:
            self.split_stereo_audios_into_two_file_by_tags()
        else:
            file_index = 0
            for tag_file_name, audio_file_name in self.tagging_mapping_audio_name.items():
                print(tag_file_name)
                print("running in {} file".format(file_index))
                tag_content = self.tag_zip.read(tag_file_name)
                sections = self.tags_to_sections(str(tag_content))
                print(len(sections))
                print(audio_file_name)
                audio, _ = librosa.load(audio_file_name, sr=self.sample_rate)
                # audio, _ = librosa.load(audio_file_name, sr=self.sample_rate, mono=False)
                audio = librosa.resample(audio, self.sample_rate, self.out_sample_rate)

                for section_index, section in enumerate(sections):
                    start = section["start"] * self.out_sample_rate
                    end = section["end"] * self.out_sample_rate
                    chunk = audio[int(start):int(end)]
                    # chunk = audio[0, int(start):int(end)]
                    path_name = 'n_section{}_{}'.format(file_index, section_index)
                    output_path = os.path.join(self.output_non_vocal_dir, path_name)
                    if section["vocal_tag"]:
                        path_name = 'v_section{}_{}'.format(file_index, section_index)
                        output_path = os.path.join(self.output_vocal_dir, path_name)
                    librosa.output.write_wav(output_path, chunk, self.out_sample_rate)
                file_index += 1

    def split_stereo_audios_by_tags(self):
        file_index = 0
        for tag_file_name, audio_file_name in self.tagging_mapping_audio_name.items():
            print(tag_file_name)
            print("running in {} file use stereo".format(file_index))
            tag_content = self.tag_zip.read(tag_file_name)
            sections = self.tags_to_sections(str(tag_content))
            print(len(sections))
            print(audio_file_name)
            audio, _ = librosa.load(audio_file_name, sr=self.sample_rate, mono=False)
            audio = librosa.resample(audio, self.sample_rate, self.out_sample_rate)
            for section_index, section in enumerate(sections):
                start = section["start"] * self.out_sample_rate
                end = section["end"] * self.out_sample_rate
                chunk = array([audio[0, int(start):int(end)], audio[1, int(start):int(end)]])
                path_name = 'n_section{}_{}'.format(file_index, section_index)
                output_path = os.path.join(self.output_non_vocal_dir, path_name)
                if section["vocal_tag"]:
                    path_name = 'v_section{}_{}'.format(file_index, section_index)
                    output_path = os.path.join(self.output_vocal_dir, path_name)
                librosa.output.write_wav(output_path, chunk, self.out_sample_rate)
            file_index += 1

    def split_stereo_audios_into_two_file_by_tags(self):
        file_index = 0
        for tag_file_name, audio_file_name in self.tagging_mapping_audio_name.items():
            print(tag_file_name)
            print("running in {} file use stereo into two file".format(file_index))
            tag_content = self.tag_zip.read(tag_file_name)
            sections = self.tags_to_sections(str(tag_content))
            print(len(sections))
            print(audio_file_name)
            audio, _ = librosa.load(audio_file_name, sr=self.sample_rate, mono=False)
            audio = librosa.resample(audio, self.sample_rate, self.out_sample_rate)

            for section_index, section in enumerate(sections):
                start = section["start"] * self.out_sample_rate
                end = section["end"] * self.out_sample_rate
                # left side
                chunk = audio[0, int(start):int(end)]
                path_name = 'n_section{}_{}'.format(file_index, section_index)
                output_path = os.path.join(self.output_left_non_vocal_dir, path_name)
                if section["vocal_tag"]:
                    path_name = 'v_section{}_{}'.format(file_index, section_index)
                    output_path = os.path.join(self.output_left_vocal_dir, path_name)
                librosa.output.write_wav(output_path, chunk, self.out_sample_rate)
                # right side
                chunk = audio[1, int(start):int(end)]
                path_name = 'n_section{}_{}'.format(file_index, section_index)
                output_path = os.path.join(self.output_right_non_vocal_dir, path_name)
                if section["vocal_tag"]:
                    path_name = 'v_section{}_{}'.format(file_index, section_index)
                    output_path = os.path.join(self.output_right_vocal_dir, path_name)
                librosa.output.write_wav(output_path, chunk, self.out_sample_rate)
            file_index += 1

    def resample_audio(self):
        resample_audio_list = find_files(self.resample_original_audio)
        for file_index, audio_file_name in enumerate(resample_audio_list):
            audio, _ = librosa.load(audio_file_name, sr=self.sample_rate)
            audio = librosa.resample(audio, self.sample_rate, self.out_sample_rate)
            path_name = 'v_section_{}'.format(file_index)
            output_path = os.path.join(self.resample_output_path, path_name)
            librosa.output.write_wav(output_path, audio, self.out_sample_rate)
            print("resample_audio", path_name)

    def merge_vocal_and_non_vocal_chunks(self):
        file_index = 0
        for tag_file_name, audio_file_name in self.tagging_mapping_audio_name.items():
            tag_content = self.tag_zip.read(tag_file_name)
            sections = self.tags_to_sections(str(tag_content))
            print(len(sections))
            print(audio_file_name)
            audio, _ = librosa.load(audio_file_name, sr=self.sample_rate)
            audio = librosa.resample(audio, self.sample_rate, self.out_sample_rate)
            for section_index, section in enumerate(sections):
                start = section["start"] * self.out_sample_rate
                end = section["end"] * self.out_sample_rate
                if section["vocal_tag"]:
                    path_name = 'v_with_non_section{}_{}.wav'.format(file_index, section_index)
                    output_path = os.path.join(self.output_merge_dir, path_name)
                    start = start - self.out_sample_rate * 0.2
                    end = end + self.out_sample_rate * 0.2

                    if start < 0:
                        start = 0
                    if end > len(audio):
                        end = len(audio)
                    chunk = audio[int(start):int(end)]
                    librosa.output.write_wav(output_path, chunk, self.out_sample_rate)
            file_index += 1

    def tags_to_sections(self, tag_content):
        sections = []
        tags = tag_content.split('\\n')
        for tag in tags:
            timestamps = re.findall('([0-9]+\.[0-9]+)', tag)
            if len(timestamps) == 0:
                continue
            vocal_tag = re.findall('(nosing|sing)', tag)[0]
            vocal_tag = True if vocal_tag == "sing" else False
            start = float(timestamps[0])
            end = float(timestamps[1])
            section = {'start': start, 'end': end, 'vocal_tag': vocal_tag}
            sections.append(section)
        return sections

    def retrieve_to_hadoop(self):
        if self.stereo_mode:
            self.retrieve_stereo_to_hadoop()
        else:
            indexList = {}
            h5 = h5py.File(self.output_hadoop_file, 'w')
            if self.fft_mode:
                x_dataset = h5.create_dataset('X', (0, self.fft_row, self.fft_col, 1), maxshape=(None, self.fft_row, self.fft_col, 1))
            elif self.use_mfcc:
                x_dataset = h5.create_dataset('X', (0, self.n_mfcc, self.mfcc_size, 1), maxshape=(None, self.n_mfcc, self.mfcc_size, 1))
            elif self.constantQ_mode:
                x_dataset = h5.create_dataset('X', (0, 84, 63, 1), maxshape=(None, 84, 63, 1))
            elif self.ceptrom_mode:
                x_dataset = h5.create_dataset(
                    'X', (0, self.cepsize, self.Hamming_widowsize, 1), maxshape=(None, self.cepsize, self.Hamming_widowsize, 1)
                )
            elif self.iirt_mode:
                x_dataset = h5.create_dataset('X', (0, 85, 61, 1), maxshape=(None, 85, 61, 1))
            else:
                x_dataset = h5.create_dataset('X', (0, self.sample_size, 1), maxshape=(None, self.sample_size, 1))
            if self.binary_mode:
                y_dataset = h5.create_dataset('Y', (0,), maxshape=(None,))
            elif self.autoencoder_mode:
                y_dataset = h5.create_dataset('Y', (0, self.fft_row, self.fft_col, 1), maxshape=(None, self.fft_row, self.fft_col, 1))
            else:
                y_dataset = h5.create_dataset('Y', (0, 2), maxshape=(None, 2))

            all_files = []
            for dataset in self.retrieve_dataset:
                for dir in self.retrieve_dir:
                    audio_dir = os.path.join(dataset, dir)
                    if os.path.exists(audio_dir):
                        print("Join", audio_dir)
                        all_files.extend(find_files(audio_dir))

            all_files.sort()
            print(all_files)
            print("Total file numbers ", len(all_files))
            if self.shuffle_all_files:
                shuffle(all_files)
            vocal_nums = 0
            non_vocal_nums = 0
            fragment_nums = 0
            # for file_index, file_name in enumerate(vocal_files):
            for file_index, file_name in enumerate(all_files):
                # print("current file {} {}".format(file_index, file_name))
                base_name = os.path.basename(file_name)
                vocal_labels = []
                if self.binary_mode:
                    label = 1 if base_name[0] == 'n' else 0
                else:
                    # label = [0, 1]
                    if file_name.find('non_vocal') > -1:
                        label = [1, 0]
                    else:
                        label = [0, 1]

                samples, _ = librosa.load(file_name, sr=self.out_sample_rate)
                # samples, _ = librosa.load(file_name, sr=self.out_sample_rate, mono=False)
                chunks = []
                chunks_num = 0
                while len(samples) >= self.sample_size:
                    # chunk = samples[0, :self.sample_size] - samples[1, :self.sample_size]
                    chunk = samples[:self.sample_size]
                    if self.fft_mode:
                        fft_features = librosa.stft(y=chunk, n_fft=self.n_fft)
                        fft_features = np.absolute(fft_features)
                        chunks.append(fft_features)
                    elif self.use_mfcc:
                        mfcc_features = np.array(
                            librosa.feature.mfcc(
                                y=chunk, sr=self.out_sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.mfcc_hop_size
                            )
                        )
                        chunks.append(mfcc_features)
                    elif self.constantQ_mode:
                        C = np.abs(librosa.cqt(y=chunk, sr=self.out_sample_rate))
                        chunks.append(C)
                    elif self.ceptrom_mode:
                        a = []
                        a = np.array(a)
                        chunks.append(a)
                    elif self.iirt_mode:
                        irrt_feature = np.abs(librosa.iirt(y=chunk, sr=16000, win_length=2048, tuning=self.turning))
                        chunks.append(irrt_feature)
                    else:
                        chunks.append(chunk)
                    vocal_labels.append(label)
                    if label[0] == 1:
                        if self.output_2s_wav:
                            librosa.output.write_wav(
                                os.path.join(self.test_music, 'non_vocal/n_{}.wav'.format(non_vocal_nums)), chunk, self.out_sample_rate
                            )
                        non_vocal_nums += 1
                    else:
                        if self.output_2s_wav:
                            librosa.output.write_wav(os.path.join(self.test_music, 'vocal/v_{}.wav'.format(vocal_nums)), chunk, self.out_sample_rate)
                        vocal_nums += 1
                    chunks_num += 1
                    indexList[fragment_nums] = base_name
                    fragment_nums += 1
                    samples = samples[int(self.sample_size * self.hop_size):]
                if chunks_num <= 0:
                    continue

                if self.autoencoder_mode:
                    if label[0] == 1:
                        y_dataset.resize(y_dataset.shape[0] + chunks_num, axis=0)
                    else:
                        x_dataset.resize(x_dataset.shape[0] + chunks_num, axis=0)
                else:
                    x_dataset.resize(x_dataset.shape[0] + chunks_num, axis=0)
                    y_dataset.resize(y_dataset.shape[0] + chunks_num, axis=0)
                chunks = np.array(chunks)

                # print(chunks.shape)
                # print(chunks_num)
                if self.fft_mode:
                    chunks = np.reshape(chunks, [chunks_num, self.fft_row, self.fft_col, 1])
                elif self.use_mfcc:
                    chunks = np.reshape(chunks, [chunks_num, self.n_mfcc, self.mfcc_size, 1])
                elif self.constantQ_mode:
                    chunks = np.reshape(chunks, [chunks_num, 84, 63, 1])
                elif self.ceptrom_mode:
                    chunks = np.reshape(chunks, [chunks_num, self.cepsize, self.Hamming_widowsize, 1])
                elif self.iirt_mode:
                    chunks = np.reshape(chunks, [chunks_num, 85, 61, 1])
                else:
                    chunks = np.reshape(chunks, [chunks_num, self.sample_size, 1])
                vocal_labels = np.array(vocal_labels)

                if self.autoencoder_mode:
                    if label[0] == 1:
                        y_dataset[-chunks_num:] = chunks
                    else:
                        x_dataset[-chunks_num:] = chunks
                else:
                    x_dataset[-chunks_num:] = chunks
                    y_dataset[-chunks_num:] = vocal_labels
                # print(non_vocal_nums, vocal_nums)
                # print(chunks_num)
            h5.close()
            with open(self.output_hadoop_file + "-pickList.json", 'w') as f:
                json.dump(indexList, f)

    def retrieve_stereo_to_hadoop(self):
        h5 = h5py.File(self.output_hadoop_file, 'w')
        indexList = {}
        if self.fft_mode:
            x_dataset = h5.create_dataset('X', (0, self.fft_row, self.fft_col, 2, 1), maxshape=(None, self.fft_row, self.fft_col, 2, 1))
        elif self.use_mfcc:
            x_dataset = h5.create_dataset('X', (0, self.n_mfcc, self.mfcc_size, 2, 1), maxshape=(None, self.n_mfcc, self.mfcc_size, 2, 1))
        else:
            x_dataset = h5.create_dataset('X', (0, self.sample_size, 2, 1), maxshape=(None, self.sample_size, 2, 1))
        if self.binary_mode:
            y_dataset = h5.create_dataset('Y', (0,), maxshape=(None,))
        else:
            y_dataset = h5.create_dataset('Y', (0, 2), maxshape=(None, 2))

        all_files = find_files(self.output_vocal_dir)
        all_files.extend(find_files(self.output_non_vocal_dir))
        # all_files.extend(find_files(self.output_uncertain_dir))

        # if has bootstrap
        if self.has_bootstrap:
            all_files.extend(find_files(self.boostrap_vocal_dir))
            all_files.extend(find_files(self.boostrap_non_vocal_dir))
        ###
        print(all_files)
        if self.shuffle_all_files:
            print("shuffle")
            shuffle(all_files)
        vocal_nums = 0
        non_vocal_nums = 0
        fragment_nums = 0
        vocal_files_length = len(find_files(self.output_vocal_dir))
        non_vocal_files_length = len(find_files(self.output_non_vocal_dir))
        vocal_per_hop = math.floor(vocal_files_length / 10)
        non_vocal_per_hop = math.floor(non_vocal_files_length / 10)
        print("vocal_per_hop: {}\n".format(vocal_per_hop))
        print("non_vocal_per_hop: {}\n".format(non_vocal_per_hop))

        for file_index, file_name in enumerate(all_files):
            print("current file {} {}".format(file_index, file_name))
            base_name = os.path.basename(file_name)
            vocal_labels = []
            if self.binary_mode:
                label = 1 if base_name[0] == 'n' else 0
            else:

                # uncertain
                if base_name[0] == 'n':
                    label = [1, 0]  # ori [1, 0]
                elif base_name[0] == 'v':
                    label = [0, 1]  # ori [0, 1]
                else:
                    label = [0.5, 0.5]
            samples, _ = librosa.load(file_name, sr=self.out_sample_rate, mono=False)
            chunks = []
            # chunks = [[], []]
            chunks_num = 0
            while len(samples[1]) >= self.sample_size:
                chunk = array([samples[0, :self.sample_size], samples[1, :self.sample_size]])
                if self.fft_mode:
                    fft_features = librosa.stft(y=chunk[0])
                    fft_features = np.absolute(fft_features)
                    chunks[0].append(fft_features)
                    fft_features = librosa.stft(y=chunk[1])
                    fft_features = np.absolute(fft_features)
                    chunks[1].append(fft_features)
                elif self.use_mfcc:
                    mfcc_features_left = np.array(
                        librosa.feature.mfcc(
                            y=chunk[0], sr=self.out_sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.mfcc_hop_size
                        )
                    )
                    mfcc_features_right = np.array(
                        librosa.feature.mfcc(
                            y=chunk[1], sr=self.out_sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.mfcc_hop_size
                        )
                    )
                    chunks[0].append(mfcc_features_left)
                    chunks[1].append(mfcc_features_right)
                else:
                    chunks.append(chunk)
                vocal_labels.append(label)

                if label[0] == 1:
                    if self.output_2s_wav:
                        librosa.output.write_wav(
                            os.path.join(self.test_music, 'non_vocal/n_{}.wav'.format(non_vocal_nums)), chunk, self.out_sample_rate
                        )
                    non_vocal_nums += 1
                else:
                    if self.output_2s_wav:
                        librosa.output.write_wav(os.path.join(self.test_music, 'vocal/v_{}.wav'.format(vocal_nums)), chunk, self.out_sample_rate)
                    vocal_nums += 1

                chunks_num += 1
                indexList[fragment_nums] = base_name
                fragment_nums += 1
                samples = array([samples[0, int(self.sample_size * self.hop_size):], samples[1, int(self.sample_size * self.hop_size):]])
            if chunks_num <= 0:
                continue
            x_dataset.resize(x_dataset.shape[0] + chunks_num, axis=0)
            y_dataset.resize(y_dataset.shape[0] + chunks_num, axis=0)
            chunks = np.array(chunks)
            print(chunks.shape)
            print(chunks_num)
            if self.fft_mode:
                chunks = np.reshape(chunks, [chunks_num, self.fft_row, self.fft_col, 2, 1])
            elif self.use_mfcc:
                chunks = np.reshape(chunks, [chunks_num, self.n_mfcc, self.mfcc_size, 2, 1])
            else:
                chunks = np.reshape(chunks, [chunks_num, self.sample_size, 2, 1])
            vocal_labels = np.array(vocal_labels)
            x_dataset[-chunks_num:] = chunks
            y_dataset[-chunks_num:] = vocal_labels
            print(non_vocal_nums, vocal_nums)
            print(chunks_num)
        h5.close()
        with open(self.output_hadoop_file + "-pickList.json", 'w') as f:
            json.dump(indexList, f)

    def verify_files(self):
        vocal_files = find_files(self.output_vocal_dir)
        non_vocal_files = find_files(self.output_non_vocal_dir)
        count = 0
        for file_name in vocal_files:
            y, _ = librosa.load(file_name, sr=self.out_sample_rate)
            if len(y) >= self.sample_size:
                count += 1
        print("verify {} audios clip".format(count))
        count = 0
        for file_name in non_vocal_files:
            y, _ = librosa.load(file_name, sr=self.out_sample_rate)
            if len(y) >= self.sample_size:
                count += 1
        print("verify {} audios clip".format(count))

    def tag_data(self, directory):
        directory = os.path.join(self.data, directory)
        vocal_files = find_files(os.path.join(directory, './vocal'))
        non_vocal_files = find_files(os.path.join(directory, './non_vocal'))
        print(os.path.join(directory, './vocal'))
        for file in vocal_files:
            dir = os.path.dirname(file)
            basename = os.path.basename(file)
            new_file = '{}/{}'.format(dir, 'v_' + basename)
            os.rename(file, new_file)

        for file in non_vocal_files:
            dir = os.path.dirname(file)
            basename = os.path.basename(file)
            new_file = '{}/{}'.format(dir, 'n_' + basename)
            os.rename(file, new_file)

    def random_category(self, directory):
        random.seed(datetime.now())
        directory = os.path.join(self.data, directory)

        random_test_directory = os.path.join(directory, './random/test')
        random_train_directory = os.path.join(directory, './random/train')

        files = find_files(os.path.join(directory, './train'))
        files.extend(find_files(os.path.join(directory, './test')))
        if self.shuffle_all_files:
            shuffle(files)
        shutil.rmtree(random_test_directory)
        shutil.rmtree(random_train_directory)

        os.makedirs(random_test_directory, exist_ok=True)
        os.makedirs(random_train_directory, exist_ok=True)
        for index, file in enumerate(files):
            file_name = os.path.basename(file)
            random_dir = random_test_directory
            if index > len(files) // 10:
                random_dir = random_train_directory
            new_file = os.path.join(random_dir, file_name)
            print(file, new_file)
            shutil.copyfile(file, new_file)

    def clear_random_directory(self, directory):
        directory = os.path.join(DATA_DIR, directory)
        test_dir = os.path.join(directory, './test')
        train_dir = os.path.join(directory, './train')

        shutil.rmtree(test_dir)
        shutil.rmtree(train_dir)

        os.makedirs(os.path.join(test_dir, 'vocal'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'non_vocal'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'vocal'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'non_vocal'), exist_ok=True)

    def added_noise_to_directory(self, directory):
        files = find_files(os.path.join(DATA_DIR, directory))
        for index, file in enumerate(files):
            file_name = os.path.basename(file)
            print(file)
            dir = os.path.dirname(file)
            new_file = '{}/{}'.format(dir, file_name + '_noise')
            noise_samples = Noise.output_noise_audio(file)
            librosa.output.write_wav(new_file, noise_samples, 16000)

    def added_pitch_down_to_directory(self, directory, out_dir):
        files = find_files(os.path.join(DATA_DIR, directory))
        out_dir = os.path.join(DATA_DIR, out_dir)
        for index, file in enumerate(files):
            file_name = os.path.basename(file)
            print(file_name)
            if file_name[0] != 'v' and file_name[0] != 'n':
                continue
            if file_name[0] == 'v':
                output_dir = os.path.join(out_dir, './vocal')
            else:
                output_dir = os.path.join(out_dir, 'non_vocal')
            new_file = '{}/{}'.format(output_dir, file_name + '_pitch_down.wav')

            pitch_samples = Pitch.output_pitch_audio(file)
            librosa.output.write_wav(new_file, pitch_samples, 16000)


# JamendoAudioReader().added_noise_to_directory('./jamendo/train')

# JamendoAudioReader().clear_random_directory('./jamendo/random')
#
# for i in range(1, 10):
#     jamendo = JamendoAudioReader()
#     jamendo.random_category('.')
#     jamendo.clear_random_directory('./jamendo/random')
#
#     jamendo.set_input_data_dir('./random/train')
#     jamendo.set_out_dir('./jamendo/random/train')
#     jamendo.retrieve_tags_and_audios()
#     jamendo.split_audios_by_tags()
#     jamendo.output_hadoop_file = os.path.join(jamendo.data, './hadoop-32k-random{}.h5'.format(i))
#     jamendo.retrieve_to_hadoop()
#
#     jamendo.set_input_data_dir('./random/test')
#     jamendo.set_out_dir('./jamendo/random/test')
#     jamendo.retrieve_tags_and_audios()
#     jamendo.split_audios_by_tags()
#     jamendo.output_hadoop_file = os.path.join(jamendo.data, './hadoop-32k-random-test{}.h5'.format(i))
#     jamendo.retrieve_to_hadoop()

# JamendoAudioReader().verify_files()

random.seed(1623)
jamendo_audio_reader = JamendoAudioReader()

# cut music
# jamendo_audio_reader.retrieve_tags_and_audios()
# jamendo_audio_reader.split_audios_by_tags()

# jamendo_audio_reader.merge_vocal_and_non_vocal_chunks()

# to hadoop
jamendo_audio_reader.retrieve_to_hadoop()

#90% output
# for i in range(0, 10):
#     jamendo_audio_reader = JamendoAudioReader(output_num = i)
#     jamendo_audio_reader.retrieve_to_hadoop()

# bootstrap
# for i in range(0, 10):
#     jamendo_audio_reader = JamendoAudioReader(bootstrap_num = i)
#     jamendo_audio_reader.retrieve_to_hadoop()

# resample
# jamendo_audio_reader.resample_audio()

# JamendoAudioReader().tag_data('./jamendo/train')

# JamendoAudioReader().added_pitch_down_to_directory('../data/jamendoFix/train', '../data/jamendoFix/pitchdown-train')
