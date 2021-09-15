import os
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa
import librosa.display


folder_path_list = ['F:\Knock-knock\Acoustic-Classification\\rawData\汇总_0106\\new',
                    'F:\Knock-knock\Acoustic-Classification\\rawData\同材质测试\\new'
                    ]
# new_dataset_folder = 'F:\Knock-knock\Acoustic-Classification\\rawData\wav_data'
new_dataset_folder = 'D:\knock_knock_DL\knock_xvector_dataset\sim-ym'

sr = 48000

# npy格式转为wav格式
# for folder_path in folder_path_list:
#     file_list = os.listdir(folder_path)
#     for npy_file in file_list:
#         npy_data = np.load(os.path.join(folder_path, npy_file))
#         for i in range(npy_data.shape[0]):
#             data = npy_data[i]
#             wav_name = npy_file + 'trials_' + str(i)
#             wav_name = wav_name.replace('.npy', '')
#             wav_name = wav_name + '.wav'
#             write(os.path.join(new_dataset_folder, wav_name), sr, data.astype(np.int16))  # 写入音频文件


file_list = os.listdir(new_dataset_folder)
for file in file_list:
    file_path = os.path.join(new_dataset_folder, file)
    waveData, sr = librosa.load(file_path, sr=sr)

    # 提取 MFCC feature
    mfccs = librosa.feature.mfcc(y=waveData, sr=sr, n_mfcc=40)  # 计算MFCC

    # 提取 mel spectrogram feature
    melspec = librosa.feature.melspectrogram(waveData, sr, n_fft=460, hop_length=200, n_mels=40)
    logmelspec = librosa.power_to_db(melspec)  # 转换为对数刻度

    # dB归一化
    max_logmel = np.max(np.absolute(logmelspec))
    logmelspec = 10*logmelspec/max_logmel

    # 绘制 mel 频谱图
    plt.figure()
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')  # 右边的色度条
    file = file.replace('.wav', '')
    plt.title(file)
    plt.savefig(os.path.join('figures_mfcc', file + '.png'))
    plt.close('all')

