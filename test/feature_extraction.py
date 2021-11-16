import os
import shutil

import librosa
import numpy as np
import random

from wav_denoising import denoising


def load_npy(audio_filepath, max_len):
    npy_data = np.load(audio_filepath).astype(np.float64)
    data_len = npy_data.shape[1]

    if data_len > max_len:  # 删除大于max_len的数据点
        extended_wav = npy_data[:, 0:max_len]
    else:
        extended_wav = npy_data
    return extended_wav


def extend_data(data, max_len=6192):
    if data.shape[1] > max_len:  # 删除大于max_len的数据点
        extended_data = data[:, :max_len]
    else:
        extended_data = np.hstack((data, np.zeros((data.shape[0], max_len - data.shape[1]))))
    return extended_data


def stft_transform(data):
    wav_data = data / np.max(np.abs(data)) * 32767
    extended_wav_data = extend_data(wav_data)
    return stft_calculating(extended_wav_data)


def stft_calculating(wav_data, n_fft=512, win_length=256, hop_length=64):
    linear = librosa.stft(wav_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    linear = linear.T
    mag, _ = librosa.magphase(linear)
    mag_T = mag.T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)

    normalized_spec = (mag_T - mu) / (std + 1e-5)
    return normalized_spec.T


def fbank_transform(wav_data):
    fbank = []
    for i in range(wav_data.shape[0]):
        fbank.append(fbank_calculating(wav_data[i].astype(float)))
    return np.array(fbank)


def fbank_calculating(wav_data, sampling_rate=48000, n_fft=512, win_length=256, hop_length=64, n_mels=40):
    mel_spec = librosa.feature.melspectrogram(wav_data, sampling_rate, n_fft=n_fft, win_length=win_length,
                                              hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.min)  # 转换为dB

    mu = np.mean(log_mel_spec, 0, keepdims=True)
    std = np.std(log_mel_spec, 0, keepdims=True)

    normalized_mel_spec = (log_mel_spec - mu) / (std + 1e-5)

    return normalized_mel_spec.T


if __name__ == '__main__':
    root_dir = '../Knock_dataset'
    domains = ['exp_data', 'sim_data']
    raw_data_dir = 'raw_data'
    feat_data_dir = 'feature_data/fbank_denoised_data'

    fs = 48000
    max_len = 6000

    for i in range(0, 2):
        shutil.rmtree(os.path.join(root_dir, feat_data_dir, domains[i]))
        os.mkdir(os.path.join(root_dir, feat_data_dir, domains[i]))

    for domain_ in domains:
        path = os.path.join(root_dir, raw_data_dir, domain_)
        files = os.listdir(path)
        for i, file in enumerate(files):
            feat_data = []
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                npy_data = load_npy(os.path.join(path, file), max_len=max_len)
                for data in npy_data:
                    data = data / max(data)
                    if domain_ == 'exp_data':
                        data = denoising(data, method='skimage-Visu')
                    fbank_feat = fbank_calculating(wav_data=data, sampling_rate=fs)  # fbank feature
                    # stft_feat = stft_calculating(wav_data=data)  # STFT feature

                    feat_data.append(fbank_feat)

                feat_data_npy = np.array(feat_data)
                save_path = os.path.join(root_dir, feat_data_dir, domain_, file)
                np.save(save_path, feat_data_npy)
