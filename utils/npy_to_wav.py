import numpy as np
import scipy.io.wavfile
import os
from sklearn.preprocessing import MinMaxScaler

dataset_path = '../Knock_dataset/raw_data'
wav_save_path = '../Knock_dataset/wav_data'

domains = ['exp_data', 'sim_data']

for dom in domains:
    npy_files = os.listdir(os.path.join(dataset_path, dom))
    for file in npy_files:
        sound_data = np.load(os.path.join(dataset_path, dom, file))[0]
        wav_save_name = os.path.join(wav_save_path, dom, file.replace('.npy', '.wav'))
        scipy.io.wavfile.write(wav_save_name, 48000, sound_data.astype('int16'))
