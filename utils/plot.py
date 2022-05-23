import os
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from definitions import DIAGRAM_DIR


def plot_spectrogram(x, shape=(63, 1024), title='Default'):
    plt.figure()
    plt.title(title)
    D = librosa.amplitude_to_db((x.reshape(shape).T), ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=16000, cmap=cm.jet, alpha=0.4)
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency')
    plt.xlabel("Time (sec)")
    plt.savefig(os.path.join(DIAGRAM_DIR, title + '.png'), bbox_inches='tight', facecolor='w')
    plt.close('all')


def plot_voting(voting_result_dict, total, title):
    sorted_items = sorted(voting_result_dict.items(), key=lambda x: x[1], reverse=True)[:]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Number')
    bars = plt.bar(*zip(*sorted_items), width=0.8)
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height/total*100:.1f}%', ha='center', va='bottom', fontsize=14)
    fig.savefig(os.path.join(DIAGRAM_DIR, title + '.png'), bbox_inches='tight', facecolor='w')
    plt.close('all')
