import numpy as np


class arg_list:
    def __init__(self, fs, n_fft, win_len, hop_len, n_mels, window='None'):
        self.fs = fs
        self.n_fft = n_fft
        self.win_len = win_len
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.window = window


def load_npy(audio_filepath, max_len, interval, max_norm=False):
    audio_data = np.load(audio_filepath).astype(np.float64)
    data_len = audio_data.shape[1]

    if data_len > max_len:  # 删除大于max_len的数据点
        audio_data = audio_data[:, 0:max_len]
    else:
        audio_data = audio_data

    audio_data = audio_data[:, int(data_len * interval[0]): int(data_len * interval[1])]

    if max_norm:
        audio_data = audio_data / np.max(audio_data)

    return audio_data
