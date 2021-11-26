import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import shutil

from utils.audio_data_load import load_npy, arg_list
from utils.wav_denoising import denoising
from model_dann_1_xvec.dataset import src_tgt_intersection


def spec_calc(audio_data, feat_type, kargs):
    if feat_type == 'stft':
        # STFT calculation
        linear_spec = librosa.stft(audio_data, n_fft=kargs.n_fft, win_length=kargs.win_len, hop_length=kargs.hop_len, window=kargs.window)
        mag_spec, _ = librosa.magphase(linear_spec)
        mag_spec_db = librosa.amplitude_to_db(mag_spec, ref=np.max)

        # STFT normalization
        mu = np.mean(mag_spec, 0, keepdims=True)
        std = np.std(mag_spec, 0, keepdims=True)
        normalized_mag_spec = (mag_spec - mu) / (std + 1e-5)

        # 频率轴变量
        freq_upper_limit = kargs.fs / 2
        f = kargs.fs * np.array(range(int(1 + kargs.n_fft / 2))) / (kargs.n_fft / 2)
        index = np.where(f < freq_upper_limit)  # 设定频率范围
        end = index[0][-1]
        f = f[:end]
        # 时间轴变量
        t = np.array(range(int(audio_data.shape[0] / kargs.hop_len + 1))) / kargs.fs

        # return t, f, normalized_mag_spec[0:end, :]

        return mag_spec_db

    elif feat_type == 'fbank':
        # 提取 mel spectrogram feature
        mel_spec = librosa.feature.melspectrogram(audio_data, kargs.fs, n_fft=kargs.n_fft,
                                                  win_length=kargs.win_len, hop_length=kargs.hop_len,
                                                  n_mels=kargs.n_mels)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # 转换为dB

        # dB归一化
        # max_log_mel = np.max(np.absolute(log_mel_spec))
        # norm_log_mel_spec = 10 * log_mel_spec/max_log_mel

        return log_mel_spec


if __name__ == '__main__':
    shutil.rmtree('../results/output_spec')
    os.mkdir('../results/output_spec')

    root_dir = '../Knock_dataset'
    raw_data_dir = 'raw_data'

    sound_npy_files = src_tgt_intersection(os.path.join(root_dir, raw_data_dir))
    domains = ['exp_data', 'sim_data', 'sim_data_aug']

    max_len = 6000
    interval = [0.0, 1.0]

    feat_type = 'stft'
    deno_method = 'pywt'  # (skimage-Visu, skimage-Bayes, pywt)
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40, window='hamming')

    y_axis_type = 'linear'

    for index, sound_npy_file in enumerate(sound_npy_files):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        if len(sound_npy_file.split('-')) < 3:
            continue

        # 实际声音
        exp_sound = load_npy(audio_filepath=os.path.join(root_dir, raw_data_dir, domains[0], sound_npy_file),
                             max_len=max_len,
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
                             max_len=max_len,
                             interval=interval)
        sim_sound_aug = load_npy(audio_filepath=os.path.join(root_dir, raw_data_dir, domains[2], sound_npy_file),
                                 max_len=max_len,
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

        fig.colorbar(img, ax=ax, format="%+2.f dB")
        fig.suptitle(sound_npy_file)
        plt.savefig(os.path.join('../results/output_spec', sound_npy_file.split('.')[0]) + '.png')
        plt.cla()
        plt.close('all')

        sys.stdout.write('\r image processing: %d / %d, %f %% ' %
                         (index, len(sound_npy_files), index/len(sound_npy_files) * 100))
        sys.stdout.flush()













