import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import os
import math
import numpy as np

from sklearn.utils import shuffle


def get_label_material(name):
    if '-wood-' in name:
        return np.int64(0)
    elif '-lv-' in name:
        return np.int64(1)
    elif '-tie-' in name:
        return np.int64(2)
    elif '-pvc-' in name:
        return np.int64(3)
    elif '-ykl-' in name:
        return np.int64(4)

    if 'changmuban' in name:  # wood
        return np.int64(0)
    elif 'duanmuban' in name:  # wood
        return np.int64(0)
    elif 'lvban' in name:  # lv
        return np.int64(1)
    elif 'tieban' in name:  # tie
        return np.int64(2)
    elif 'xiaozhuodi' in name:  # wood
        return np.int64(0)
    elif 'xiaozhuoding' in name:  # wood
        return np.int64(0)
    elif 'xiaozhuozuo' in name:  # wood
        return np.int64(0)
    elif 'yinxiangbei' in name:  # pvc
        return np.int64(3)
    elif 'yinxiangdi' in name:  # wood
        return np.int64(0)
    elif 'yinxiangding' in name:  # wood
        return np.int64(0)
    elif 'yinxiangyou' in name:  # wood
        return np.int64(0)
    elif 'yklban' in name:  # ykl
        return np.int64(4)
    elif 'zhuogeban' in name:  # wood
        return np.int64(0)
    elif 'duanmuban-lv' in name:  # wood
        return np.int64(1)
    elif 'duanmuban-tie' in name:  # wood
        return np.int64(2)
    elif 'duanmuban-ykl' in name:  # wood
        return np.int64(3)
    else:
        print('Error!')
        return np.int64(0)


def get_label_object(name):
    if len(name.split('-')) > 3:
        return np.int64(int(name.split('-')[0]) + 12)

    if 'changmuban' in name:  # wood
        return np.int64(0)
    elif 'duanmuban' in name:  # wood
        return np.int64(1)
    elif 'lvban' in name:  # lv
        return np.int64(2)
    elif 'tieban' in name:  # tie
        return np.int64(3)
    elif 'xiaozhuoding' in name:  # wood
        return np.int64(4)
    elif 'xiaozhuodi' in name:  # wood
        return np.int64(5)
    elif 'xiaozhuozuo' in name:  # wood
        return np.int64(6)
    elif 'yinxiangbei' in name:  # pvc
        return np.int64(7)
    elif 'yinxiangding' in name:  # wood
        return np.int64(8)
    elif 'yinxiangdi' in name:  # wood
        return np.int64(9)
    elif 'yinxiangyou' in name:  # wood
        return np.int64(10)
    elif 'yklban' in name:  # ykl
        return np.int64(11)
    elif 'zhuogeban' in name:  # wood
        return np.int64(12)
    return np.int64(13)


def get_label_object2(name):
    if 'changmuban-' in name:  # wood
        return np.int64(0)
    elif 'duanmuban-' in name:  # wood
        return np.int64(1)
    elif 'lvban-' in name:  # lv
        return np.int64(2)
    elif 'tieban-' in name:  # tie
        return np.int64(3)
    elif 'xiaozhuoding-' in name:  # wood
        return np.int64(4)
    elif 'yinxiangdi-' in name:  # wood
        return np.int64(5)
    elif 'yinxiangding-' in name:  # wood
        return np.int64(6)
    elif 'yklban-' in name:  # ykl
        return np.int64(7)
    elif 'zhuogeban-' in name:  # wood
        return np.int64(8)
    return np.int64(9)


def balanced_sampling_on_src_tgt(root_dir, file_name):
    src_sample = np.load(os.path.join(root_dir, 'exp_data', file_name)).astype(np.float32)
    tgt_sample = np.load(os.path.join(root_dir, 'sim_data', file_name)).astype(np.float32)

    balan_ratio = (src_sample.shape[0] / tgt_sample.shape[0])

    if balan_ratio > 1:
        tgt_sample = np.repeat(tgt_sample, math.ceil(balan_ratio), axis=0)
    else:
        src_sample = np.repeat(src_sample, math.ceil(1/balan_ratio), axis=0)

    return src_sample, tgt_sample


def src_tgt_intersection(root_dir):
    # 取模拟数据和实际数据的共同部分
    src_files = os.listdir(os.path.join(root_dir, 'exp_data'))
    tgt_files = os.listdir(os.path.join(root_dir, 'sim_data'))
    common_files = [file for file in src_files if file in tgt_files]

    # 过滤common数据
    common_files = [file for file in common_files if len(file.split('-')) > 3]

    return common_files


def load_data(root_dir, domain, train_flag):
    '''
    :param root_dir: 数据集根目录
    :param domain: 数据集域，exp for 实际敲击数据，sim for 模拟数据
    :param train_flag: load_data读取数据的使用目的，为1表示训练（需要src-tgt均衡），为0表示测试（不需要src-tgt均衡）
    :return: 随机排序后的特征和标签
    '''
    shuffle_random_state = 20

    # 过滤common数据
    common_files = src_tgt_intersection(root_dir=root_dir)

    # 读取 + 合并数据
    if train_flag == 1:
        x_data_list, y_data_list = [], []
        for i, file in enumerate(common_files):
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                src_data, tgt_data = balanced_sampling_on_src_tgt(root_dir=root_dir, file_name=file_name)
                if domain == 'exp_data':
                    np_data = src_data
                else:
                    np_data = tgt_data
                for data in np_data:
                    x_data_list.append(data)
                    y_data_list.append(get_label_object(name))
    elif train_flag == 0:
        path = os.path.join(root_dir, domain)
        x_data_list, y_data_list = [], []
        for i, file in enumerate(common_files):
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                np_data = np.load(os.path.join(path, file_name)).astype(np.float32)
                for data in np_data:
                    # data = data[:, :28, :28]
                    # data[:, :, 17:28] = 0
                    x_data_list.append(data)
                    y_data_list.append(get_label_object(name))

    x_data_total = np.array(x_data_list)
    y_data_total = np.array(y_data_list)
    # x_data_total = x_data_total[y_data_total != 13]
    # y_data_total = y_data_total[y_data_total != 13]
    x_data_total, y_data_total = shuffle(x_data_total, y_data_total, random_state=shuffle_random_state)  # 打乱数据顺序
    return x_data_total, y_data_total


class KnockDataset_train(Dataset):
    '''
    训练集包括源域和目标域除support set以外的所有类别的所有样本
    '''
    def __init__(self, root_dir, domain, support_label_set):
        self.x_data_total, self.y_data_total = load_data(root_dir, domain, train_flag=1)  # 读取磁盘数据
        self.total_label_set = set(self.y_data_total)
        self.train_label_set = list(self.total_label_set - support_label_set)  # 剔除支撑集label
        self.train_label_set.sort()
        self.train_data = np.empty((0, self.x_data_total.shape[1], self.x_data_total.shape[2]), dtype=np.float32)
        self.train_labels = np.empty((0,), np.int32)
        self.n_classes = len(self.train_label_set)

        for i in iter(self.train_label_set):
            num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
            self.train_data = np.vstack((self.train_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
            self.train_labels = np.hstack((self.train_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))

        self.train_data = torch.Tensor(self.train_data)
        self.train_labels = torch.Tensor(self.train_labels)

    def __getitem__(self, index):
        img = self.train_data[index: index + 1]
        label = int(self.train_labels[index: index + 1])
        return img, label

    def __len__(self):
        return len(self.train_data)


class KnockDataset_test(Dataset):
    '''
    测试集的样本是support set中所有类别在源域中的样本
    '''
    def __init__(self, root_dir, domain, support_label_set):
        self.x_data_total, self.y_data_total = load_data(root_dir, domain, train_flag=0)  # 读取磁盘数据
        self.test_data = np.empty((0, self.x_data_total.shape[1], self.x_data_total.shape[2]), dtype=np.float32)
        self.test_labels = np.empty((0,), np.int32)
        self.n_classes = len(support_label_set)

        support_label_set = list(support_label_set)
        support_label_set.sort()

        for i in iter(support_label_set):
            num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
            self.test_data = np.vstack((self.test_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
            self.test_labels = np.hstack((self.test_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))

        ### 将label重新调整为连续整数
        # for index, label in enumerate(support_label_set):
        #     self.test_labels[self.test_labels == label] = index

        self.test_data = torch.Tensor(self.test_data)
        self.test_labels = torch.Tensor(self.test_labels)

    def __getitem__(self, index):
        wav = self.test_data[index: index + 1]
        label = int(self.test_labels[index: index + 1])
        return wav, label

    def __len__(self):
        return len(self.test_data)


class KnockDataset_pair(Dataset):
    def __init__(self, root_dir, support_label_set):
        domain_list = ['exp_data', 'sim_data']
        exp_data_dir = os.path.join(root_dir, domain_list[0])
        sim_data_dir = os.path.join(root_dir, domain_list[1])

        self.total_npy_files = src_tgt_intersection(root_dir=root_dir)

        self.total_exp_x_data = []
        self.total_sim_x_data = []

        for npy_file in self.total_npy_files:
            name, postfix = npy_file.split('.')
            if get_label_object(name) in support_label_set:
                continue

            exp_x_data, sim_x_data = balanced_sampling_on_src_tgt(root_dir=root_dir, file_name=npy_file)

            for exp_data in exp_x_data:
                self.total_exp_x_data.append(exp_data)
            for sim_data in sim_x_data:
                self.total_sim_x_data.append(sim_data)

        self.total_exp_x_data = torch.Tensor(np.array(self.total_exp_x_data, dtype=np.float32))
        self.total_sim_x_data = torch.Tensor(np.array(self.total_sim_x_data, dtype=np.float32))

    def __getitem__(self, item):
        exp_data = self.total_exp_x_data[item: item + 1]
        sim_data = self.total_exp_x_data[item: item + 1]
        return exp_data, sim_data

    def __len__(self):
        return self.total_exp_x_data.shape[0]


# class KnockDataset(Dataset):
#     def __init__(self, root_dir, domain, train=False):
#         self.n_classes = 13
#         self.train = train
#         self.test_ratio = 0.2
#
#         # 生成训练集
#         if self.train:
#             self.x_data_total, self.y_data_total = load_data(root_dir, domain)  # 读取磁盘数据
#             self.train_data = np.empty((0, self.x_data_total.shape[1], self.x_data_total.shape[2]), dtype=np.float32)
#             self.train_labels = np.empty((0,), np.int32)
#
#             for i in range(self.n_classes):
#                 num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
#                 num_of_data = int(num_of_data * (1-self.test_ratio))
#                 self.train_data = np.vstack((self.train_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
#                 self.train_labels = np.hstack((self.train_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))
#
#             self.train_data = torch.Tensor(self.train_data)
#             self.train_labels = torch.Tensor(self.train_labels)
#         # 生成测试集
#         else:
#             self.x_data_total_source, self.y_data_total_source = load_data(root_dir, domain[0])  # 读取实验数据
#             self.x_data_total_target, self.y_data_total_target = load_data(root_dir, domain[1])  # 读取模拟数据
#             self.test_data = np.empty((0, self.x_data_total_source.shape[1], self.x_data_total_source.shape[2]), dtype=np.float32)
#             self.test_labels = np.empty((0,), np.int32)
#
#             for i in range(self.n_classes):
#                 num_of_data = self.y_data_total_source[self.y_data_total_source == i].shape[0]
#                 num_of_data = int(num_of_data * (1-self.test_ratio))
#                 self.test_data = np.vstack(
#                     (self.test_data, self.x_data_total_source[self.y_data_total_source == i][num_of_data:]))
#                 self.test_labels = np.hstack(
#                     (self.test_labels, self.y_data_total_source[self.y_data_total_source == i][num_of_data:]))
#
#             for i in range(self.n_classes):
#                 num_of_data = self.y_data_total_target[self.y_data_total_target == i].shape[0]
#                 num_of_data = int(num_of_data * (1-self.test_ratio))
#                 self.test_data = np.vstack(
#                     (self.test_data, self.x_data_total_target[self.y_data_total_target == i][num_of_data:]))
#                 self.test_labels = np.hstack(
#                     (self.test_labels, self.y_data_total_target[self.y_data_total_target == i][num_of_data:]))
#
#             self.test_data = torch.Tensor(self.test_data)
#             self.test_labels = torch.Tensor(self.test_labels)
#
#     def __getitem__(self, index):
#         if self.train:
#             img = self.train_data[index: index + 1]
#             label = int(self.train_labels[index: index + 1])
#         else:
#             img = self.test_data[index: index + 1]
#             label = int(self.test_labels[index: index + 1])
#         return img, label
#
#     def __len__(self):
#         if self.train:
#             return len(self.train_data)
#         else:
#             return len(self.test_data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        # while self.count + self.batch_size < self.n_dataset:
        while 1:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size


# def support_set_generation(root_dir, domain, support_label_set):
#     path = os.path.join(root_dir, domain)
#     x_data_list, y_data_list = [], []
#     files = os.listdir(path)
#     for i, file in enumerate(files):
#         file_name = str(file)
#         name, postfix = file_name.split('.')
#         if postfix == 'npy':
#             np_data = np.load(os.path.join(path, file_name)).astype(np.float32)
#             for data in np_data:
#                 data_label = get_label_object(name)
#                 if data_label in support_label_set:
#                     x_data_list.append(data)
#                     y_data_list.append(get_label_object(name))
#
#     x_data_support = torch.Tensor(np.array(x_data_list))
#     y_data_support = torch.Tensor(np.array(y_data_list))
#
#     # self.train_data = np.vstack((self.train_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
#     # self.train_labels = np.hstack((self.train_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))
#
#     return x_data_support, y_data_support


if __name__ == '__main__':
    root_dir = 'Knock_dataset/stft_noisy_data'
    domain = 'sim_data'

    # train_dataset = KnockDataset(root_dir, domain, train=True)
    # test_dataset = KnockDataset(root_dir, domain, train=False)
    # dataset = KnockDataset_train(root_dir, domain, {0, 3, 5})
    # support_set_generation(root_dir, domain, {0, 3, 5}, 3)
    # n_classes = train_dataset.n_classes

    dataset = KnockDataset_pair(root_dir, support_label_set=[])
    dataset.__getitem__(8)
