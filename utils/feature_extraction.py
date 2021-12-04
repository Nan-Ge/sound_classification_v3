import os
import shutil
import sys

import librosa
import numpy as np

from audio_preprocessing.wav_denoising import denoising
from utils.audio_data_load import arg_list, load_npy


def feat_calc(audio_data, kargs):
    if kargs.feat_type == 'stft':
        linear = librosa.stft(
            audio_data,
            n_fft=kargs.n_fft,
            win_length=kargs.win_len,
            hop_length=kargs.hop_len,
            window=kargs.window
        )
        mag, _ = librosa.magphase(linear)

        mu = np.mean(mag, 0, keepdims=True)
        std = np.std(mag, 0, keepdims=True)
        normalized_spec = (mag - mu) / (std + 1e-5)

        return normalized_spec

    elif kargs.feat_type == 'fbank':
        mel_spec = librosa.feature.melspectrogram(
            audio_data,
            kargs.fs,
            n_fft=kargs.n_fft,
            win_length=kargs.win_len,
            hop_length=kargs.hop_len,
            n_mels=kargs.n_mels,
            window=kargs.window
        )

        mu = np.mean(mel_spec, 0, keepdims=True)
        std = np.std(mel_spec, 0, keepdims=True)
        normalized_mel_spec = (mel_spec - mu) / (std + 1e-5)

        return normalized_mel_spec


def feat_extraction(root_data_dir, feat_data_dir, kargs):
    raw_data_dir = 'raw_data'
    doms = ['exp_data', 'sim_data', 'sim_data_aug']

    for i in range(len(doms)):
        if os.path.exists(os.path.join(root_data_dir, feat_data_dir, doms[i])):
            return 0
        shutil.rmtree(os.path.join(root_data_dir, feat_data_dir, doms[i]), ignore_errors=True)
        os.makedirs(os.path.join(root_data_dir, feat_data_dir, doms[i]))

    max_len = kargs.max_len
    interval = [kargs.interval, 1.0]
    deno_method = kargs.deno_method  # (skimage-Visu, skimage-Bayes, pywt)

    for domain_ in doms:
        path = os.path.join(root_data_dir, raw_data_dir, domain_)
        files = os.listdir(path)
        for i, file in enumerate(files):
            feat_data = []
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                npy_data = load_npy(os.path.join(path, file), max_len=max_len, interval=interval)
                for data in npy_data:
                    # data = data / max(data)

                    if domain_ == 'exp_data':
                        data = denoising(data, method=deno_method)

                    feat = feat_calc(audio_data=data, kargs=kargs)
                    feat_data.append(feat)

                feat_data_npy = np.array(feat_data)
                save_path = os.path.join(root_data_dir, feat_data_dir, domain_, file)
                np.save(save_path, feat_data_npy)


if __name__ == '__main__':
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40, window='hann')
    feat_extraction(
        feat_data_dir='feature_data/stft_whole',
        kargs=kargs
    )
