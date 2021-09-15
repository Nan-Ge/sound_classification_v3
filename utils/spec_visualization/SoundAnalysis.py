import SPAnalysis
import FeatureExtraction
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import filename_parsing
from minmaxnormalization import minmaxnormalizer
from Finding_sound import find_sound

###################################################  频谱分析 FFT  #####################################################
mat = ['ykl']
obj_id = ['1']
knock_obj = ['ph2']
recorder = ['mic2','ph1']
strength = ['n']
micro_pos = ['side']
sup_method = ['di1']
knocker = ['a']
time = ['0106a']
sig_len = ['len3']
knock_pos = ['3']


val_name_dict = {'mat': mat,
                 'obj_id': obj_id,
                 'knock_obj': knock_obj,
                 'recorder': recorder,
                 'strength': strength,
                 'micro_pos': micro_pos,
                 'sup_method': sup_method,
                 'knocker': knocker,
                 'time': time,
                 'sig_len': sig_len,
                 'knock_pos': knock_pos}

val_name_list = list(val_name_dict.keys())

res, wavedata = find_sound(folder_path='F:\Knock-knock\Acoustic-Classification\\rawData\汇总_0106\\new',
                           mat=mat,
                           obj_id=obj_id,
                           knock_obj=knock_obj,
                           recorder=recorder,
                           strength=strength,
                           micro_pos=micro_pos,
                           sup_method=sup_method,
                           knocker=knocker,
                           time=time,
                           sig_len=sig_len,
                           knock_pos=knock_pos)

# file_list = [
#     'dm-1-sj-ph2-n-side-di1-a-1229e-len2-3-.npy',
#     'dm-1-sj-ph2-n-side-di1-a-1229e-len2-5-.npy',
#     'dm-1-sj-ph2-n-side-di1-a-1229e-len2-7-.npy',
#     'dm-1-ph2-ph1-n-side-di1-b-1216e-len1-3-.npy',
#     'dm-1-ph2-ph1-n-side-di1-b-1216e-len1-5-.npy',
#     'dm-1-ph2-ph1-n-side-di1-b-1216e-len1-9-.npy'
# ]
# folder_path = 'F:\Knock-knock\Acoustic-Classification\\rawData\汇总_1229'
# wavedata = np.zeros(shape=(1, 6000))
# for index, file_name in enumerate(file_list):
#     if os.path.exists(os.path.join(folder_path, file_name)):
#         wavedata_temp = np.load(os.path.join(folder_path, file_name))
#         # 对采样时间短的样本进行补齐
#         if wavedata_temp.shape[1] == 3000:
#             wavedata_temp = np.concatenate((np.zeros(shape=(wavedata_temp.shape[0], 1500)), wavedata_temp), axis=1)
#             wavedata_temp = np.concatenate((wavedata_temp, np.zeros(shape=(wavedata_temp.shape[0], 1500))), axis=1)
#         wavedata = np.concatenate((wavedata, wavedata_temp), axis=0)
#     else:
#         del file_list[index]
# wavedata = np.delete(wavedata, 0, axis=0)


fft_actu, fft_freq = SPAnalysis.SpecAnalysis(wavedata, Method='FFT', sound_source='acu',
                                             sampling_rate=48000, freq_thres=8000)
fft_actu = minmaxnormalizer(fft_actu)

condition = ''
for i in range(len(val_name_list)):
    if len(val_name_dict[val_name_list[i]]) > 1:
        condition += val_name_list[i].upper() + '-'
        # if val_name_list[i] is not 'knock_pos':
        compare_val = val_name_dict[val_name_list[i]]  # 待比较的变量
    else:
        condition += val_name_dict[val_name_list[i]][0] + '-'

color_list = ['b', 'r', 'g', 'y']
color_english = ['Blue','Red','Green','Yellow']
plt.figure(figsize=(50, 20))

# 同时比较三个位置
# for index_i, obj_i in enumerate(knock_pos):
#     plt.subplot(len(knock_pos), 1, index_i+1)
#     # plt.text()
#     if index_i == 0:
#         plt.title(condition, fontsize='xx-large', fontweight='bold')
#     for index_j, obj_j in enumerate(compare_val):
#         for t in range(10):
#             plt.plot(fft_freq, fft_actu[(10 * (index_i + index_j * len(knock_pos)) + t), :], color=color_list[index_j])  # 绘图
#             plt.ylim(0, 1.2)

# 设置图例
annotation = ''
for i in range(len(compare_val)):
    temp = compare_val[i] + '-' + color_english[i]
    annotation = annotation + temp + '\n'
plt.title(condition, fontsize='xx-large', fontweight='bold')

# 控制变量唯一 + 无平均
# for index_j, obj_j in enumerate(compare_val):
#     for t in range(10):
#         plt.plot(fft_freq, fft_actu[10*index_j+t, :], color=color_list[index_j])  # 绘图
#         plt.ylim(0, 1.2)

# 控制变量唯一 + 平均10次
average_fft = np.zeros(shape=(len(compare_val), fft_actu.shape[1]))
for i in range(len(compare_val)):
    temp = np.mean(fft_actu[i*10:i*10+10, :], axis=0)
    average_fft[i, :] = temp[np.newaxis, :]
for index_j, obj_j in enumerate(compare_val):
    plt.plot(fft_freq, average_fft[index_j], color=color_list[index_j])  # 绘图
    plt.ylim(0, 1.2)


plt.text(6000, 0.6, annotation, size=60, alpha=0.8)
plt.savefig(condition + '.png')
plt.close('all')

#######################################################################################################################

# acu_position_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']  # 实际声音的敲击位置
# simu_position_list = ['3', '5', '7']  # 模拟声音的敲击位置
#
# folder_path = 'rawData/12.1-morning'
# ds_name = 'wood2X131-12'
# names = os.listdir(os.path.join(folder_path, ds_name))
# del names[names.index('sen')]  # 将sen文件夹从names列表中删除
#
# exp_para = filename_parsing.parser(ds_name)  # 解析实验数据收集环境
# condition = exp_para[5] + '-' + exp_para[2]
#
# ft_flag = 0  # ndarray初始化标志
#
# names = [
#     'F:\Knock-knock\Acoustic-Classification\\test\dm-1-ph1-n-side-b-3.npy'
#     ,'F:\Knock-knock\Acoustic-Classification\\test\dm-1-ph1-n-side-b-1.npy'
#     # ,'F:\Knock-knock\Acoustic-Classification\\test\ym-1-ph1-wyz-n-side-3.npy'
# ]
#
# # 'F:\Knock-knock\Acoustic-Classification\\test\dm-1-ph1-wyz-n-side-1.npy'
# ##### 实际声音的频谱计算
# for name in names:
#     ID = re.findall(r"\d", name)  # 正则表达式提取样本实验时的参数
#     # if ID[1] in acu_position_list:  # ID[1]表征的是敲击位置，ID[1]的类型为str
#     npy_path = os.path.join(folder_path, ds_name, name)  # 读取npy数据矩阵
#     fft_actu, fft_freq = SPAnalysis.SpecAnalysis(npy_path)  # 计算实际敲击声音FFT
#      # fft_avg_actu = np.mean(fft_actu, axis=0)  # 对20个样本的FFT结果进行平均
#     if ft_flag == 0:
#         fft_all_actu = np.zeros(shape=(1, fft_actu.shape[1]))
#         ft_flag = 1
#     fft_all_actu = np.vstack((fft_all_actu, fft_actu))
#
# fft_all_actu = np.delete(fft_all_actu, 0, axis=0)
# # fft_all_actu = minmaxnormalizer(fft_all_actu)
#
# ##### 模拟声音的频谱计算
# # folder_path = 'rawData'
# # ds_name = 'simulation_result_1'
# # ft_flag = 0
# #
# # names = os.listdir(os.path.join(folder_path, ds_name))
# # for name in names:
# #     if name.split('.')[-1] == 'npy':
# #         ID = re.findall(r"\d", name)
# #         if ID[0] in simu_position_list:
# #             npy_path = os.path.join(folder_path, ds_name, name)
# #             fft_simu, fft_freq = SPAnalysis.SpecAnalysis(npy_path, flag=1)  # 计算模拟声音FFT
# #             if ft_flag == 0:
# #                 fft_all_simu = np.zeros(shape=(1, fft_simu.shape[1]))
# #                 ft_flag = 1
# #             fft_all_simu = np.vstack((fft_all_simu, fft_simu))
# #
# # fft_all_simu = np.delete(fft_all_simu, 0, axis=0)
# # fft_all_simu = minmaxnormalizer(fft_all_simu)
#
# ##### 绘图
# # plt.figure(figsize=(32, 32))
# # for j in range(1, len(acu_position_list)+1):
# #     plt.subplot(len(acu_position_list), 1, j)
# #     if j == 1:
# #         plt.title(condition, fontsize='xx-large', fontweight='bold')
# #     # 真实声音
# #     for t in range(20):
# #         plt.plot(fft_freq, fft_all_actu[(20*j+t-20)], 'b')  # 绘图
# #         plt.ylim(0, 1.2)
# #     # 模拟声音
# #     # plt.plot(fft_freq, fft_all_simu[j-1], 'r')
# # plt.savefig('FFT_' + condition + '.png')
# # plt.show()
#
# plt.figure(figsize=(50, 10))
# for j in range(1, 30):
#     # 真实声音
#     if j < 10:
#         plt.plot(fft_freq, fft_all_actu[j], 'b')  # 绘图
#         # plt.ylim(0, 1.2)
#     elif 10 <= j < 20:
#         plt.plot(fft_freq, fft_all_actu[j], 'r')  # 绘图
#         # plt.ylim(0, 1.2)
#     # else:
#     #     plt.plot(fft_freq, fft_all_actu[j], 'g')  # 绘图
#     #     # plt.ylim(0, 1.2)
#
#     # 模拟声音
#     # plt.plot(fft_freq, fft_all_simu[j-1], 'r')
# plt.savefig('FFT.png')
# plt.show()

# ##################################################  特征分析  #########################################################
# ##### 实际敲击声音特征提取
# acu_position_list = ['3', '5', '7']  # 感兴趣的位置列表
# i = 0  # 记录当前已进行到第(i+1)个感兴趣的位置
#
# folder_path = 'rawData/12.1-morning'
# ds_name = 'wood2X131-12'
# names = os.listdir(os.path.join(folder_path, ds_name))
# del names[names.index('sen')]  # 将sen文件夹从names列表中删除
#
# for name in names:
#     ID = re.findall(r"\d", name)  # 正则表达式提取样本实验时的参数
#     if ID[1] in acu_position_list:  # ID[1]表征的是敲击位置，ID[1]的类型为str
#         npy_path = os.path.join(folder_path, ds_name, name)  # 读取npy数据矩阵
#         sound_feature = FeatureExtraction.SoundFeatureExtraction(npy_path)  # 提取声学特征
#
#         if i == 0:
#             shape_1 = (sound_feature.shape[0] * len(acu_position_list), sound_feature.shape[1])
#             feature_matrix = np.zeros(shape_1, dtype=float)  # 存储所有样本的特征
#             shape_2 = (sound_feature.shape[0] * len(acu_position_list), 1)
#             label_matrix = np.zeros(shape_2, dtype=float)  # 存储所有样本的label
#
#         feature_matrix[i*20:i*20+20, :] = sound_feature
#         label_matrix[i*20:i*20+20] = np.ones((20, 1), dtype=float)*int(ID[1])
#
#         i = i + 1
#
#
# ##### 模拟声音特征提取
# folder_path = 'rawData'
# ds_name = 'simulation_result_1'
# ft_flag = 0
#
# simu_position_list = ['3', '5', '7']  # 感兴趣的位置列表
#
# names = os.listdir(os.path.join(folder_path, ds_name))
# for name in names:
#     if name.split('.')[-1] == 'npy':
#         ID = re.findall(r"\d", name)
#         if ID[0] in simu_position_list:
#             npy_path = os.path.join(folder_path, ds_name, name)
#             sound_feature = FeatureExtraction.SoundFeatureExtraction(npy_path)  # 提取声学特征
#
#             feature_matrix = np.vstack((feature_matrix, sound_feature))
#             label_matrix = np.vstack((label_matrix, np.ones((1, 1), dtype=float)*int(ID[0])*10))
#
# # PCA降维
# pca_estimator = PCA(n_components=0.85)
# pca_feature_matrix = pca_estimator.fit_transform(feature_matrix[0:60, :])
#
#
# # t-SNE可视化
# tsne_estimator = TSNE(n_components=2)
# tsne_feature_matrix = tsne_estimator.fit_transform(pca_feature_matrix)
#
# x_min, x_max = np.min(tsne_feature_matrix, 0), np.max(tsne_feature_matrix, 0)
# norm_tsne_feature_matrix = (tsne_feature_matrix - x_min) / (x_max - x_min)
#
# # 绘制降维后的数据图片
# plt.figure(figsize=(12, 12))
# for i in range(norm_tsne_feature_matrix.shape[0]):
#     plt.text(norm_tsne_feature_matrix[i, 0], norm_tsne_feature_matrix[i, 1], str(label_matrix[i]), color=plt.cm.Set1(int(label_matrix[i])/10.), fontdict={'weight': 'bold', 'size': 9})
# plt.savefig('tsne.png')
# plt.show()

