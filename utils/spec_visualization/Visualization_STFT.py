import SPAnalysis
import FeatureExtraction
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import filename_parsing
from minmaxnormalization import minmaxnormalizer
from Finding_sound import find_sound

folder_path = 'F:\Knock-knock\Acoustic-Classification\\rawData\simulation_result_1'
file_list = os.listdir(folder_path)

for file in file_list:
    file_path = os.path.join(folder_path, file)
    waveData = np.load(file_path)
    waveData = waveData.astype(np.float)

    stft_t, stft_f, stft_spec = SPAnalysis.SpecAnalysis(waveData,
                                                        Method='STFT',
                                                        sound_source='sim',
                                                        sampling_rate=48000,
                                                        freq_thres=12000)

    for i in range(stft_spec.shape[0]):
        plt.figure(figsize=(16, 10))
        plt.pcolormesh(stft_t, stft_f, stft_spec[i])
        plt.colorbar()
        plt.title(file + '_' + str(i))
        plt.ylabel('F')
        plt.xlabel('t')
        plt.tight_layout()
        plt.savefig(os.path.join('stft_figures', file + '_' + str(i) + '.png'))
        plt.close('all')
