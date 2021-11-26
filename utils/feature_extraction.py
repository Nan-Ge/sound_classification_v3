import os
import shutil

import librosa
import numpy as np
import random

from utils.wav_denoising import denoising
from utils.audio_data_load import arg_list, load_npy


def feat_calc(audio_data, kargs, feat_type):
    if feat_type == 'stft':
        linear = librosa.stft(audio_data, n_fft=kargs.n_fft, win_length=kargs.win_len, hop_length=kargs.hop_len)
        linear = linear.T
        mag, _ = librosa.magphase(linear)
        mag_T = mag.T

        mu = np.mean(mag_T, 0, keepdims=True)
        std = np.std(mag_T, 0, keepdims=True)
        normalized_spec = (mag_T - mu) / (std + 1e-5)

        return normalized_spec.T

    elif feat_type == 'fbank':
        mel_spec = librosa.feature.melspectrogram(audio_data, kargs.fs, n_fft=kargs.n_fft, win_length=kargs.win_len,
                                                  hop_length=kargs.hop_len, n_mels=kargs.n_mels)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.min)  # 转换为dB

        mu = np.mean(log_mel_spec, 0, keepdims=True)
        std = np.std(log_mel_spec, 0, keepdims=True)
        normalized_mel_spec = (log_mel_spec - mu) / (std + 1e-5)

        return normalized_mel_spec.T


if __name__ == '__main__':
    root_dir = '../Knock_dataset'
    domains = ['exp_data', 'sim_data_aug']
    raw_data_dir = 'raw_data'
    feat_data_dir = 'feature_data/fbank_denoised_data'

    for i in range(0, 2):
        shutil.rmtree(os.path.join(root_dir, feat_data_dir, domains[i]))
        os.mkdir(os.path.join(root_dir, feat_data_dir, domains[i]))

    max_len = 6000
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40)
    interval = [0.4, 1.0]

    feat_type = 'fbank'
    deno_method = 'pywt'  # (skimage-Visu, skimage-Bayes, pywt)

    for domain_ in domains:
        path = os.path.join(root_dir, raw_data_dir, domain_)
        files = os.listdir(path)
        for i, file in enumerate(files):
            feat_data = []
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                npy_data = load_npy(os.path.join(path, file), max_len=max_len, interval=interval)
                for data in npy_data:
                    data = data / max(data)
                    if domain_ == 'exp_data':
                        data = denoising(data, method=deno_method)

                    feat = feat_calc(audio_data=data, kargs=kargs, feat_type=feat_type)  # fbank feature
                    feat_data.append(feat)

                feat_data_npy = np.array(feat_data)
                save_path = os.path.join(root_dir, feat_data_dir, domain_, file)
                np.save(save_path, feat_data_npy)
