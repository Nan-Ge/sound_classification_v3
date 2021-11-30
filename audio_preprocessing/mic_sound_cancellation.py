import os.path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import scipy.io.wavfile
import noisereduce as nr

from utils.spec_visualization import spec_calc
from utils.audio_data_load import arg_list, load_npy, am_normalization

from audio_preprocessing.pyaec.time_domain_adaptive_filters.lms import lms
from audio_preprocessing.pyaec.time_domain_adaptive_filters.nlms import nlms
from audio_preprocessing.pyaec.time_domain_adaptive_filters.blms import blms
from audio_preprocessing.pyaec.time_domain_adaptive_filters.bnlms import bnlms
from audio_preprocessing.pyaec.time_domain_adaptive_filters.rls import rls
from audio_preprocessing.pyaec.time_domain_adaptive_filters.apa import apa
from audio_preprocessing.pyaec.time_domain_adaptive_filters.kalman import kalman
from audio_preprocessing.pyaec.frequency_domain_adaptive_filters.pfdaf import pfdaf
from audio_preprocessing.pyaec.frequency_domain_adaptive_filters.fdaf import fdaf
from audio_preprocessing.pyaec.frequency_domain_adaptive_filters.fdkf import fdkf
from audio_preprocessing.pyaec.frequency_domain_adaptive_filters.pfdkf import pfdkf
from audio_preprocessing.pyaec.nonlinear_adaptive_filters.volterra import svf
from audio_preprocessing.pyaec.nonlinear_adaptive_filters.flaf import flaf
from audio_preprocessing.pyaec.nonlinear_adaptive_filters.aeflaf import aeflaf
from audio_preprocessing.pyaec.nonlinear_adaptive_filters.sflaf import sflaf
from audio_preprocessing.pyaec.nonlinear_adaptive_filters.cflaf import cflaf


def index_trans(src, fig_size):
    return [int(src/fig_size[1]), int(src % fig_size[1])]


def spec_plot(subplot_size, save_name):
    plt.rcParams['savefig.dpi'] = 800  # 图片像素
    plt.rcParams['figure.dpi'] = 800  # 分辨率
    fig, ax = plt.subplots(nrows=subplot_size[0], ncols=subplot_size[1], figsize=(20, 20))

    for index in range(subplot_size[0] * subplot_size[1]):
        sub_index = index_trans(index, subplot_size)
        sub_fig = ax[sub_index[0], sub_index[1]]
        sub_fig.get_xaxis().set_visible(False)
        sub_fig.get_yaxis().set_visible(False)
        # sub_fig.label_outer()

    for index, audio_data in enumerate(audio_list):
        spec = spec_calc(audio_data=audio_data, kargs=kargs, feat_type=feat_type)
        sub_index = index_trans(index, subplot_size)
        sub_fig = ax[sub_index[0], sub_index[1]]
        img = librosa.display.specshow(spec, y_axis=y_axis, x_axis='time', sr=kargs.fs, ax=sub_fig)

        sub_fig.set(title=method_list[index])

    fig.colorbar(img, ax=ax, format="%+2.f")
    fig.suptitle("Different AEC algorithm.")
    plt.savefig(os.path.join(spec_save_path, save_name))
    plt.cla()


def self_vib_cancellation_AEC(mixAudio, pureAudio):

    fs = kargs.fs
    audio_arr = []

    # ---------------------- time domain adaptive filters -----------------------
    e = np.clip(lms(pureAudio, mixAudio, N=256, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(blms(pureAudio, mixAudio, N=256, L=4, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(nlms(pureAudio, mixAudio, N=256, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(bnlms(pureAudio, mixAudio, N=256, L=4, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(rls(pureAudio, mixAudio, N=256), -1, 1)
    audio_arr.append(e)

    e = np.clip(apa(pureAudio, mixAudio, N=256, P=5, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(kalman(pureAudio, mixAudio, N=256), -1, 1)
    audio_arr.append(e)

    # -------------------------- nonlinear adaptive filters -----------------------
    e = np.clip(svf(pureAudio, mixAudio, M=256, mu1=0.1, mu2=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(flaf(pureAudio, mixAudio, M=256, P=5, mu=0.2), -1, 1)
    audio_arr.append(e)

    e = np.clip(aeflaf(pureAudio, mixAudio, M=256, P=5, mu=0.05, mu_a=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(sflaf(pureAudio, mixAudio, M=256, P=5, mu_L=0.2, mu_FL=0.5), -1, 1)
    audio_arr.append(e)

    e = np.clip(cflaf(pureAudio, mixAudio, M=256, P=5, mu_L=0.2, mu_FL=0.5, mu_a=0.5), -1, 1)
    audio_arr.append(e)

    # ---------------------- frequency domain adaptive filters --------------------
    e = np.clip(fdaf(pureAudio, mixAudio, M=256, mu=0.1), -1, 1)
    audio_arr.append(e)

    e = np.clip(fdkf(pureAudio, mixAudio, M=256), -1, 1)
    audio_arr.append(e)

    e = np.clip(pfdaf(pureAudio, mixAudio, N=8, M=64, mu=0.1, partial_constrain=True), -1, 1)
    audio_arr.append(e)

    e = np.clip(pfdkf(pureAudio, mixAudio, N=8, M=64, partial_constrain=True), -1, 1)
    audio_arr.append(e)

    return audio_arr


def self_vib_cancellation_noiReduce(mixAudio, pureAudio):
    audio_arr = []
    # Stationary
    e = nr.reduce_noise(
        y=mixAudio,
        sr=kargs.fs,
        stationary=True,
        chunk_size=512,
        n_fft=kargs.n_fft,
    )
    audio_arr.append(e)

    # Non - stationary
    e = nr.reduce_noise(
        y=mixAudio,
        sr=kargs.fs,
        y_noise=pureAudio,
        stationary=False,
        chunk_size=512,
        n_fft=kargs.n_fft,
    )
    audio_arr.append(e)

    return audio_arr


if __name__ == '__main__':
    # ------------- Global Paras ----------------
    wav_save_path = os.path.join('result', 'wav')
    spec_save_path = os.path.join('result', 'spec')

    feat_type = 'stft'
    deno_method = 'pywt'  # (skimage-Visu, skimage-Bayes, pywt)
    kargs = arg_list(fs=48000, n_fft=256, win_len=256, hop_len=64, n_mels=40, window='hanning')

    max_len = 6000
    interval = [0.0, 1.0]
    y_axis = 'linear'

    # --------------------------------------------

    mix_audio = load_npy(audio_filepath='mix.npy', max_len=max_len, interval=interval, am_norm=False)[0]
    pure_audio = load_npy(audio_filepath='pure.npy', max_len=max_len, interval=interval, am_norm=False)[0]

    scipy.io.wavfile.write(os.path.join(wav_save_path, 'mix.wav'), 48000, mix_audio.astype('int16'))
    scipy.io.wavfile.write(os.path.join(wav_save_path, 'pure.wav'), 48000, mix_audio.astype('int16'))

    # ------------ Alg 1: AEC -------------
    method_list = [
        'lms', 'blms', 'nlms', 'bnlms', 'rls', 'apa', 'kalman',
        'svf', 'flaf', 'aeflaf', 'sflaf', 'cflaf',
        'fdaf', 'fdkf', 'pfdaf', 'pfdkf',
        'MIX', 'PURE', 'MIC',
    ]
    audio_list = self_vib_cancellation_AEC(
        mixAudio=am_normalization(mix_audio),
        pureAudio=am_normalization(pure_audio),
    )

    # -------- Alg 2: Noise Reduce ---------
    # method_list = [
    #     'Stationary', 'Non-stationary',
    #     'MIX', 'PURE'
    # ]
    # audio_list = self_vib_cancellation_noiReduce(
    #     mixAudio=am_normalization(mix_audio),
    #     pureAudio=am_normalization(pure_audio),
    # )

    audio_list.append(mix_audio)
    audio_list.append(pure_audio)

    spec_plot(subplot_size=(4, 5), save_name='AEC_result')
