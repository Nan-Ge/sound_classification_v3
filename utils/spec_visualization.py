import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import shutil

from utils.audio_data_load import load_npy, arg_list
from audio_preprocessing.wav_denoising import denoising
from model_dann_1_xvec.dataset import src_tgt_intersection


def spec_calc(audio_data, feat_type, kargs):
    if feat_type == 'stft':
        # STFT calculation
        linear_spec = librosa.stft(audio_data, n_fft=kargs.n_fft, win_length=kargs.win_len, hop_length=kargs.hop_len, window=kargs.window)
        mag_spec, _ = librosa.magphase(linear_spec)
        # mag_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)

        # STFT normalization
        mu = np.mean(mag_spec, axis=0, keepdims=True)
        std = np.std(mag_spec, axis=0, keepdims=True)
        normalized_mag_spec = (mag_spec - mu) / (std + 1e-5)

        return normalized_mag_spec

    elif feat_type == 'fbank':
        # Mel_spec calculation
        mel_spec = librosa.feature.melspectrogram(
            audio_data,
            kargs.fs,
            n_fft=kargs.n_fft,
            win_length=kargs.win_len, hop_length=kargs.hop_len,
            n_mels=kargs.n_mels)
        # mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为dB

        # Mel_spec normalization
        mu = np.mean(mel_spec, 0, keepdims=True)
        std = np.std(mel_spec, 0, keepdims=True)
        normalized_mel_spec = (mel_spec - mu) / (std + 1e-5)

        return normalized_mel_spec


if __name__ == '__main__':
    shutil.rmtree('../results/output_spec')
    os.mkdir('../results/output_spec')

    root_dir = '../Knock_dataset'
    raw_data_dir = 'raw_data'

    sound_npy_files = src_tgt_intersection(os.path.join(root_dir, raw_data_dir), dom=['exp_data', 'sim_data'])
    domains = ['exp_data', 'sim_data', 'sim_data_aug']

    max_len = 6000
    interval = [0.4, 1.0]

    feat_type = 'stft'
    y_axis_type = 'linear'
    deno_method = 'skimage-Bayes'  # (skimage-Visu, skimage-Bayes, pywt)
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40, window='hann')

    for index, sound_npy_file in enumerate(sound_npy_files):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        if len(sound_npy_file.split('-')) < 3:
            continue

        # 实际声音
        exp_sound = load_npy(audio_filepath=os.path.join(root_dir, raw_data_dir, domains[0], sound_npy_file),
                             new_len=max_len,
                             interval=interval)

        spec = spec_calc(audio_data=exp_sound[0], kargs=kargs, feat_type=feat_type)
        img = librosa.display.specshow(spec, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[0, 0])
        ax[0, 0].set(title='exp-raw')
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0, 0].label_outer()

        spec = spec_calc(audio_data=denoising(exp_sound[0], method=deno_method), kargs=kargs, feat_type=feat_type)
        librosa.display.specshow(spec, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[1, 0])
        ax[1, 0].set(title='exp-deno')
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1, 0].label_outer()

        # 模拟声音
        sim_sound = load_npy(audio_filepath=os.path.join(root_dir, raw_data_dir, domains[1], sound_npy_file),
                             new_len=max_len,
                             interval=interval)
        sim_sound_aug = load_npy(audio_filepath=os.path.join(root_dir, raw_data_dir, domains[2], sound_npy_file),
                                 new_len=max_len,
                                 interval=interval)

        spec = spec_calc(audio_data=sim_sound[0], kargs=kargs, feat_type=feat_type)
        librosa.display.specshow(spec, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[0, 1])
        ax[0, 1].set(title='sim-raw')
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0, 1].label_outer()

        spec = spec_calc(audio_data=sim_sound_aug[0], kargs=kargs, feat_type=feat_type)
        librosa.display.specshow(spec, y_axis=y_axis_type, x_axis='time', sr=kargs.fs, ax=ax[1, 1])
        ax[1, 1].set(title='sim-aug')
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].label_outer()

        fig.colorbar(img, ax=ax, format="%+2.f")
        fig.suptitle(sound_npy_file)
        plt.savefig(os.path.join('../results/output_spec', sound_npy_file.split('.')[0]) + '.png')
        plt.cla()
        plt.close('all')

        sys.stdout.write('\r image processing: %d / %d, %f %% ' %
                         (index, len(sound_npy_files), index/len(sound_npy_files) * 100))
        sys.stdout.flush()













