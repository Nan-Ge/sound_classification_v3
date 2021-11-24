import librosa
import numpy as np


def extend_data(data, max_len=6144):
    if data.shape[1] > max_len:  # 删除大于max_len的数据点
        extended_data = data[:, :max_len]
    else:
        extended_data = np.hstack((data, np.zeros((data.shape[0], max_len - data.shape[1]))))
    return extended_data


def stft_transform(data):
    wav_data = extend_data(data)
    stft_result = []
    for i in range(wav_data.shape[0]):
        stft_result.append(fbank_calculating(stft_result[i].astype(float)))
    return np.array(stft_result)


def stft_calculating(wav_data, n_fft=512, win_length=256, hop_length=64):
    linear = librosa.stft(wav_data, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    linear = linear.T
    mag, _ = librosa.magphase(linear)
    mag_T = mag.T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)

    normalized_spec = (mag_T - mu) / (std + 1e-5)
    return normalized_spec.T


def fbank_transform(data):
    wav_data = extend_data(data)
    fbank_result = []
    for i in range(wav_data.shape[0]):
        fbank_result.append(fbank_calculating(wav_data[i].astype(float)))
    return np.array(fbank_result)


def fbank_calculating(wav_data, sampling_rate=48000, n_fft=512, win_length=256, hop_length=64, n_mels=40):
    mel_spec = librosa.feature.melspectrogram(wav_data, sampling_rate, n_fft=n_fft, win_length=win_length,
                                              hop_length=hop_length, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.min)  # 转换为dB

    mu = np.mean(log_mel_spec, 0, keepdims=True)
    std = np.std(log_mel_spec, 0, keepdims=True)

    normalized_mel_spec = (log_mel_spec - mu) / (std + 1e-5)

    return normalized_mel_spec.T
