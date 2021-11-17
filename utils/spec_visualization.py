import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import shutil

from feature_extraction import load_npy
from wav_denoising import denoising
from model_dann_xvec.dataset import src_tgt_intersection


def stft_visualization(sound_data, sampling_rate, n_fft=512, win_length=256, hop_length=64):
    # STFT calculation
    linear_spec = librosa.stft(sound_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mag_spec, _ = librosa.magphase(linear_spec)
    log_mag_spec = librosa.amplitude_to_db(mag_spec, ref=np.max)

    # STFT normalization
    mu = np.mean(mag_spec, 0, keepdims=True)
    std = np.std(mag_spec, 0, keepdims=True)
    normalized_mag_spec = (mag_spec - mu) / (std + 1e-5)

    # 频率轴变量
    freq_upper_limit = sampling_rate / 2
    f = sampling_rate * np.array(range(int(1 + n_fft / 2))) / (n_fft / 2)
    index = np.where(f < freq_upper_limit)  # 设定频率范围
    end = index[0][-1]
    f = f[:end]
    # 时间轴变量
    t = np.array(range(int(sound_data.shape[0] / hop_length + 1))) / sampling_rate

    # return t, f, normalized_mag_spec[0:end, :]

    return log_mag_spec


def fbank_visualization(sound_data, sampling_rate, n_fft=512, win_length=256, hop_length=64, n_mels=40):
    # 提取 mel spectrogram feature
    melspec = librosa.feature.melspectrogram(sound_data, sampling_rate, n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(melspec, ref=np.max)  # 转换为dB


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

    domains = ['exp_data', 'sim_data']

    deno_method = 'pywt'

    max_len = 6000
    fs = 48000

    for index, sound_npy_file in enumerate(sound_npy_files):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        if len(sound_npy_file.split('-')) < 3:
            continue

        ### 实际声音
        exp_sound = load_npy(os.path.join(root_dir, raw_data_dir, domains[0], sound_npy_file), max_len)

        spec = fbank_visualization(sound_data=exp_sound[0], sampling_rate=fs)
        img = librosa.display.specshow(spec, y_axis='mel', x_axis='time', sr=fs, ax=ax[0, 0])
        ax[0, 0].set(title='exp-noisy-fbank')
        ax[0, 0].get_xaxis().set_visible(False)
        ax[0, 0].label_outer()

        spec = fbank_visualization(sound_data=denoising(exp_sound[0], method=deno_method), sampling_rate=fs)
        librosa.display.specshow(spec, y_axis='mel', x_axis='time', sr=fs, ax=ax[1, 0])
        ax[1, 0].set(title='exp-deno-fbank')
        ax[1, 0].get_xaxis().set_visible(False)
        ax[1, 0].label_outer()


        ### 模拟声音
        sim_sound = load_npy(os.path.join(root_dir, raw_data_dir, domains[1], sound_npy_file), max_len)

        spec = fbank_visualization(sound_data=sim_sound[0], sampling_rate=fs)
        librosa.display.specshow(spec, y_axis='mel', x_axis='time', sr=fs, ax=ax[0, 1])
        ax[0, 1].set(title='sim-noisy-fbank')
        ax[0, 1].get_xaxis().set_visible(False)
        ax[0, 1].label_outer()

        spec = fbank_visualization(sound_data=denoising(sim_sound[0], method=deno_method), sampling_rate=fs)
        librosa.display.specshow(spec, y_axis='mel', x_axis='time', sr=fs, ax=ax[1, 1])
        ax[1, 1].set(title='sim-deno-fbank')
        ax[1, 1].get_xaxis().set_visible(False)
        ax[1, 1].label_outer()

        # spec = stft_visualization(sound_data=denoising(exp_sound[0], method='pywt'), sampling_rate=fs)  # 频谱计算
        # librosa.display.specshow(spec, y_axis='linear', x_axis='time', sr=fs, ax=ax[3])
        # ax[3].set(title='Soft_threshold')
        # ax[3].label_outer()

        ### 实际声音 Mel-Spec
        # mel_spec = mfcc_visualization(sound_data=exp_sound[0], sampling_rate=fs)
        #
        # librosa.display.specshow(mel_spec, y_axis='mel', x_axis='time', sr=fs, ax=ax[1])
        # ax[1].set(title='Mel-Spec')
        # ax[1].label_outer()
        #
        # fig.colorbar(img, ax=ax, format="%+2.f dB")

        ### 模拟声音 STFT
        # sim_sound = load_npy(os.path.join(root_dir, raw_data_dir, domains[1], sound_npy_file), max_len)
        # sim_t, sim_f, sim_spec = stft_visualization(sim_sound[0])

        # plt.subplot(1, 2, 2)
        # plt.pcolormesh(sim_t, sim_f, sim_spec)
        # plt.colorbar()
        # plt.title(sound_npy_file + '_sim_stft')
        # plt.ylabel('Frequency')
        # plt.xlabel('Frame')
        # plt.tight_layout()

        fig.colorbar(img, ax=ax, format="%+2.f dB")
        fig.suptitle(sound_npy_file)
        plt.savefig(os.path.join('../results/output_spec', sound_npy_file.split('.')[0]) + '.png')
        plt.cla()
        # plt.close('all')

        sys.stdout.write('\r image processing: %d / %d, %f %% ' % (index, len(sound_npy_files), index/len(sound_npy_files) * 100))
        sys.stdout.flush()













