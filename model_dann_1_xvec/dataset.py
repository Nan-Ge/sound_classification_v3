import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from model_dann_1_xvec.config import *

import os
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


def get_label_obj_face(file_name):

    res = LABEL_DICT.get('-'.join(file_name.split('-')[0:2]), "Not Found!")
    if res == "Not Found!":
        print(file_name)
        return np.int64(0)
    else:
        return res


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


def src_tgt_intersection(root_dir):
    # 取模拟数据和实际数据的共同部分
    src_files = os.listdir(os.path.join(root_dir, 'exp_data'))
    tgt_files = os.listdir(os.path.join(root_dir, 'sim_data'))
    common_files = [file for file in src_files if file in tgt_files]

    # 过滤common数据
    common_files = [file for file in common_files if len(file.split('-')) > 3]

    return common_files


def balanced_sampling_on_src_tgt(root_dir, file_name):
    src_sample = np.load(os.path.join(root_dir, 'exp_data', file_name)).astype(np.float32)
    tgt_sample = np.load(os.path.join(root_dir, 'sim_data', file_name)).astype(np.float32)

    balan_dif = src_sample.shape[0] - tgt_sample.shape[0]

    if balan_dif > 0:
        rand_slice = np.random.choice(range(0, tgt_sample.shape[0]), size=balan_dif, replace=True)
        tgt_sample = np.concatenate((tgt_sample, tgt_sample[rand_slice]), axis=0)
    elif balan_dif < 0:
        rand_slice = np.random.choice(range(0, src_sample.shape[0]), size=-balan_dif, replace=True)
        src_sample = np.concatenate((src_sample, src_sample[rand_slice]), axis=0)
    else:
        pass

    return src_sample, tgt_sample


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
        src_x_list, src_y_list = [], []
        tgt_x_list, tgt_y_list = [], []

        for i, file in enumerate(common_files):
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                src_data, tgt_data = balanced_sampling_on_src_tgt(root_dir=root_dir, file_name=file_name)
                if domain == 'exp_data':
                    np_data = src_data
                elif domain == 'sim_data':
                    np_data = tgt_data

                for data in src_data:
                    src_x_list.append(data)
                    src_y_list.append(get_label_obj_face(name))

                for data in tgt_data:
                    tgt_x_list.append(data)
                    tgt_y_list.append(get_label_obj_face(name))

        src_x_total = np.array(src_x_list)
        src_y_total = np.array(src_y_list, dtype=np.int64)
        tgt_x_total = np.array(tgt_x_list)
        tgt_y_total = np.array(tgt_y_list, dtype=np.int64)

        src_x_total, src_y_total = shuffle(src_x_total, src_y_total, random_state=shuffle_random_state)  # 打乱数据顺序
        tgt_x_total, tgt_y_total = shuffle(tgt_x_total, tgt_y_total, random_state=shuffle_random_state)  # 打乱数据顺序

        return (src_x_total, src_y_total), (tgt_x_total, tgt_y_total)

    elif train_flag == 0:
        path = os.path.join(root_dir, domain)
        x_data_list, y_data_list = [], []
        for i, file in enumerate(common_files):
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                np_data = np.load(os.path.join(path, file_name)).astype(np.float32)
                for data in np_data:
                    x_data_list.append(data)
                    y_data_list.append(get_label_obj_face(name))

        x_data_total = np.array(x_data_list)
        y_data_total = np.array(y_data_list)

        x_data_total, y_data_total = shuffle(x_data_total, y_data_total, random_state=shuffle_random_state)  # 打乱数据顺序
        return x_data_total, y_data_total


class KnockDataset_train(Dataset):
    '''
    训练集包括源域和目标域除support set以外的所有类别的（所有样本 * （1-val_ratio））
    '''
    def __init__(self, root_data, support_label_set, val_ratio=0.2):
        self.x_total = root_data[0]
        self.y_total = root_data[1]

        self.total_label_set = set(self.y_total)
        self.train_label_set = list(self.total_label_set - support_label_set)  # 剔除支撑集label
        self.train_label_set = list(self.total_label_set)
        self.train_label_set.sort()

        self.train_data = np.empty((0, self.x_total.shape[1], self.x_total.shape[2]), dtype=np.float32)
        self.train_label = np.empty((0,), np.int32)

        self.n_classes = len(self.train_label_set)

        for i in iter(self.train_label_set):
            num_of_train_data = int(self.y_total[self.y_total == i].shape[0] * (1 - val_ratio))
            self.train_data = np.vstack((self.train_data, self.x_total[self.y_total == i][:num_of_train_data]))
            self.train_label = np.hstack((self.train_label, self.y_total[self.y_total == i][:num_of_train_data]))

        self.train_data = torch.Tensor(self.train_data)
        self.train_label = torch.Tensor(self.train_label)

    def __getitem__(self, index):
        img = self.train_data[index: index + 1]
        label = int(self.train_label[index: index + 1])
        return img, label

    def __len__(self):
        return len(self.train_data)


class KnockDataset_val(Dataset):
    '''
    训练集包括源域和目标域除support set以外的所有类别的（所有样本 * val_ratio）
    '''

    def __init__(self, root_data, support_label_set, val_ratio=0.2):
        self.x_total = root_data[0]
        self.y_total = root_data[1]

        self.total_label_set = set(self.y_total)
        self.val_label_set = list(self.total_label_set - support_label_set)  # 剔除支撑集label
        self.val_label_set = list(self.total_label_set)
        self.val_label_set.sort()

        self.val_data = np.empty((0, self.x_total.shape[1], self.x_total.shape[2]), dtype=np.float32)
        self.val_label = np.empty((0,), np.int32)

        self.n_classes = len(self.val_label_set)

        for i in iter(self.val_label_set):
            num_of_train_data = int(self.y_total[self.y_total == i].shape[0] * (1 - val_ratio))

            self.val_data = np.vstack((self.val_data, self.x_total[self.y_total == i][num_of_train_data:]))
            self.val_label = np.hstack((self.val_label, self.y_total[self.y_total == i][num_of_train_data:]))

        self.val_data = torch.Tensor(self.val_data)
        self.val_label = torch.Tensor(self.val_label)

    def __getitem__(self, index):
        img = self.val_data[index: index + 1]
        label = int(self.val_label[index: index + 1])
        return img, label

    def __len__(self):
        return len(self.val_data)


class KnockDataset_test(Dataset):
    '''
    测试集的样本是support set中所有类别在源域中的样本
    '''
    def __init__(self, root_dir, domain, support_label_set):
        self.x_data_total, self.y_data_total = load_data(root_dir, domain, train_flag=0)  # 读取磁盘数据
        self.test_data = np.empty((0, self.x_data_total.shape[1], self.x_data_total.shape[2]), dtype=np.float32)
        self.test_labels = np.empty((0,), np.int32)
        self.n_classes = len(support_label_set)

        if len(support_label_set) == 0:
            self.test_data = self.x_data_total
            self.test_labels = self.y_data_total
        else:
            support_label_set = list(support_label_set)
            support_label_set.sort()

            for i in iter(support_label_set):
                num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
                self.test_data = np.vstack((self.test_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
                self.test_labels = np.hstack(
                    (self.test_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))

            # 将label重新调整为连续整数
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
    def __init__(self, src_root_data, tgt_root_data, support_label_set, val_ratio=0.2):
        self.src_x_total = src_root_data[0]
        self.src_y_total = src_root_data[1]
        self.tgt_x_total = tgt_root_data[0]
        self.src_y_total = tgt_root_data[1]

        # 剔除支撑集label
        self.total_label_set = set(self.src_y_total)
        self.pair_label_set = list(self.total_label_set - support_label_set)
        self.pair_label_set = list(self.total_label_set)
        self.pair_label_set.sort()

        # 获取label数量
        self.n_classes = len(self.pair_label_set)

        # 提取数据
        self.exp_data = np.empty((0, self.src_x_total.shape[1], self.src_x_total.shape[2]), dtype=np.float32)
        self.exp_label = np.empty((0,), np.int64)
        self.sim_data = np.empty((0, self.tgt_x_total.shape[1], self.tgt_x_total.shape[2]), dtype=np.float32)
        self.sim_label = np.empty((0,), np.int64)

        for i in iter(self.pair_label_set):
            train_data_num = int(self.src_y_total[self.src_y_total == i].shape[0] * (1 - val_ratio))

            self.exp_data = np.vstack((self.exp_data, self.src_x_total[self.src_y_total == i][:train_data_num]))
            self.exp_label = np.concatenate(
                (self.exp_label, self.src_y_total[self.src_y_total == i][:train_data_num]))

            self.sim_data = np.vstack((self.sim_data, self.tgt_x_total[self.src_y_total == i][:train_data_num]))
            self.sim_label = np.concatenate(
                (self.sim_label, self.src_y_total[self.src_y_total == i][:train_data_num]))

        self.exp_data = torch.Tensor(self.exp_data, )
        self.exp_label = torch.Tensor(self.exp_label).long()
        self.sim_data = torch.Tensor(self.sim_data)
        self.sim_label = torch.Tensor(self.sim_label).long()

    def __getitem__(self, item):
        exp_data = self.exp_data[item: item + 1]
        exp_label = self.exp_label[item: item + 1]
        sim_data = self.sim_data[item: item + 1]
        sim_label = self.sim_label[item: item + 1]
        return (exp_data, exp_label), (sim_data, sim_label)

    def __len__(self):
        return self.exp_data.shape[0]


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


if __name__ == '__main__':
    root_dir = '../Knock_dataset/feature_data/stft_noisy_data'
    domain = 'sim_data'

    # train_dataset = KnockDataset(root_dir, domain, train=True)
    # test_dataset = KnockDataset(root_dir, domain, train=False)
    # dataset = KnockDataset_train(root_dir, domain, {0, 3, 5})
    # support_set_generation(root_dir, domain, {0, 3, 5}, 3)
    # n_classes = train_dataset.n_classes

    dataset = KnockDataset_pair(root_dir, support_label_set=[])
    dataset.__getitem__(8)
