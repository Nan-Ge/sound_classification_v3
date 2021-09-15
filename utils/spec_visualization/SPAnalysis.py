import pylab as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import pywt
import librosa
import wav_deno
from sklearn.preprocessing import normalize
from minmaxnormalization import minmaxnormalizer

def SpecAnalysis(waveData, Method = 'FFT', sound_source = 'acu', sampling_rate = 48000, freq_thres = 5000):
    # 相关参数设置
    t = np.arange(0, 1.0, 1.0 / sampling_rate)  # 波形持续时间

    # 读取npy数据矩阵
    if sound_source == 'acu':
        fft_size = waveData.shape[1]
        # if waveData.shape[1] == 3000:
        #     # waveData = np.hstack((waveData[:waveData.shape[0]], np.zeros((waveData.shape[0], 72))))  # np.hstack将参数元组的元素数组按水平方向进行叠加，在这里将3000个采样点补齐到3072
        #     fft_size = 3000  # fft_size实际上等于采样点的个数
        # elif waveData.shape[1] == 6000:
        #     fft_size = 6000
    else:
        # waveData = waveData[0:3072]
        waveData = waveData[np.newaxis,:]

    # 振幅归一化
    # waveData = normalize(waveData, axis=1, norm='max')
    # amplitude_min_max_scalar = preprocessing.MinMaxScaler(feature_range=(-1, 1))  # 创建振幅最大最小归一化对象
    # waveData = amplitude_min_max_scalar.fit_transform(waveData)
    # waveData = minmaxnormalizer(waveData)

    # 小波降噪
    # waveData = wav_deno.wave_denosing(wavedata=waveData, wave_let_func='sym4', maxlev=5, wav_threshold=0.3)

    # 计算filename指向音频的FFT
    if Method == 'FFT':
        xf = np.apply_along_axis(np.fft.rfft, 1, waveData) / fft_size  # 计算音频样本的FFT
        freqs = np.linspace(0, sampling_rate / 2, fft_size // 2 + 1)  # 频率轴自变量
        xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))  # 第一种情况，求得的FFT做阈值限定
        xfp2 = np.abs(xf)  # 第二种情况，求得的FFT不做阈值限定

        index = np.where(freqs<freq_thres)  # 设定频率范围
        end = index[0][-1]
        truncated_freq = freqs[:end]

        return xfp2[:, :end], truncated_freq

    elif Method == 'STFT':
        hop_length = 160
        n_fft = 512
        win_length = 400
        i = 0
        for i in range(waveData.shape[0]):
            linear = librosa.stft(waveData[i], n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
            if i == 0:
                stft_spec = np.zeros(shape=(waveData.shape[0], linear.shape[0], linear.shape[1]))
            stft_spec[i] = np.abs(linear)

        f = sampling_rate * np.array(range(int(1 + n_fft / 2))) / (n_fft / 2)
        t = np.array(range(int(waveData.shape[1]/hop_length+1)))/sampling_rate

        index = np.where(f < freq_thres)  # 设定频率范围
        end = index[0][-1]
        truncated_f = f[:end]

        return t, truncated_f, np.abs(stft_spec)[:,0:end,:]


    # 计算filename指向音频的CWT
    elif Method == 'CWT':
        # 小波参数的设置
        wavename = 'cgau8'  # 小波名称
        totalscal = 480  # 尺度参数
        fc = pywt.central_frequency(wavename)  # 小波函数的中心频率
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)

        # 一个npy文件中的所有样本的cwt结果存储在一起
        shape = (waveData.shape[0],totalscal-1, waveData.shape[1])
        cwt_output = np.zeros(shape,dtype=float)

        for i in range(waveData.shape[0]):
            data = waveData[i, :]
            [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)  # 进行CWT同时得到变换结果和频率轴自变量
            cwt_output[i] = abs(cwtmatr)

        return cwt_output, frequencies, t[:waveData.shape[1]]
