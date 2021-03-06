import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from model_dann_1_xvec.config import *

import os

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
        return res - 1


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


def src_tgt_intersection(dataset_dir, dom):
    # ?????????????????????????????????????????????
    src_files = os.listdir(os.path.join(dataset_dir, dom[0]))
    tgt_files = os.listdir(os.path.join(dataset_dir, dom[1]))
    common_files = [file for file in src_files if file in tgt_files]

    # ??????common??????
    common_files = [file for file in common_files if len(file.split('-')) > 3]

    return common_files


def balanced_sampling_on_src_tgt(dataset_dir, dom, file_name):
    src_sample = np.load(os.path.join(dataset_dir, dom[0], file_name)).astype(np.float32)
    tgt_sample = np.load(os.path.join(dataset_dir, dom[1], file_name)).astype(np.float32)

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


def load_data(dataset_dir, dom, train_flag):
    '''
    :param dataset_dir: ??????????????????
    :param dom: ???????????????exp for ?????????????????????sim for ????????????
    :param train_flag: load_data?????????????????????????????????1?????????????????????src-tgt???????????????0????????????????????????src-tgt?????????
    :return: ?????????????????????????????????
    '''
    shuffle_random_state = 20

    # ??????common??????
    common_files = src_tgt_intersection(dataset_dir=dataset_dir, dom=dom)

    # ?????? + ????????????
    if train_flag == 1:
        src_x_list, src_y_list = [], []
        tgt_x_list, tgt_y_list = [], []

        for i, file in enumerate(common_files):
            file_name = str(file)
            name, postfix = file_name.split('.')
            if postfix == 'npy':
                src_data, tgt_data = balanced_sampling_on_src_tgt(dataset_dir=dataset_dir, file_name=file_name, dom=dom)

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

        src_x_total, src_y_total = shuffle(src_x_total, src_y_total, random_state=shuffle_random_state)  # ??????????????????
        tgt_x_total, tgt_y_total = shuffle(tgt_x_total, tgt_y_total, random_state=shuffle_random_state)

        return (src_x_total, src_y_total), (tgt_x_total, tgt_y_total)

    elif train_flag == 0:
        path = os.path.join(dataset_dir, dom)
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

        x_data_total, y_data_total = shuffle(x_data_total, y_data_total, random_state=shuffle_random_state)  # ??????????????????
        return x_data_total, y_data_total


class KnockDataset_train(Dataset):
    '''
    ????????????????????????????????????support set??????????????????????????????????????? * ???1-val_ratio??????
    '''
    def __init__(self, root_data, support_label_set, val_ratio=0.2):
        self.x_total = root_data[0]
        self.y_total = root_data[1]

        self.total_label_set = set(self.y_total)
        self.train_label_set = list(self.total_label_set - support_label_set)  # ???????????????label
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
    ????????????????????????????????????support set??????????????????????????????????????? * val_ratio???
    '''

    def __init__(self, root_data, support_label_set, val_ratio=0.2):
        self.x_total = root_data[0]
        self.y_total = root_data[1]

        self.total_label_set = set(self.y_total)
        self.val_label_set = list(self.total_label_set - support_label_set)  # ???????????????label
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

        self.data_shape = self.val_data.shape

    def __getitem__(self, index):
        img = self.val_data[index: index + 1]
        label = int(self.val_label[index: index + 1])
        label = self.val_label_set.index(label)

        return img, label

    def __len__(self):
        return len(self.val_data)


class KnockDataset_test(Dataset):
    '''
    ?????????????????????support set????????????????????????????????????
    '''
    def __init__(self, root_data, support_label_set):
        self.x_data_total = root_data[0]
        self.y_data_total = root_data[1]
        self.test_data = np.empty((0, self.x_data_total.shape[1], self.x_data_total.shape[2]), dtype=np.float32)
        self.test_labels = np.empty((0,), np.int32)
        self.n_classes = len(support_label_set)

        if len(support_label_set) == 0:
            self.test_data = self.x_data_total
            self.test_labels = self.y_data_total
        else:
            support_label_set = list(support_label_set)
            support_label_set.sort()
            self.support_label_set = support_label_set

            for i in iter(support_label_set):
                num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
                self.test_data = np.vstack((self.test_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
                self.test_labels = np.hstack(
                    (self.test_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))

            # ???label???????????????????????????
            # for index, label in enumerate(support_label_set):
            #     self.test_labels[self.test_labels == label] = index

            self.test_data = torch.Tensor(self.test_data)
            self.test_labels = torch.Tensor(self.test_labels)

            self.data_shape = self.test_data.shape

    def __getitem__(self, index):
        wav = self.test_data[index: index + 1]
        label = int(self.test_labels[index: index + 1])
        label = self.support_label_set.index(label)

        return wav, label

    def __len__(self):
        return len(self.test_data)


class KnockDataset_pair(Dataset):
    def __init__(self, src_root_data, tgt_root_data, support_label_set, val_ratio=0.2):
        self.src_x_total = src_root_data[0]
        self.src_y_total = src_root_data[1]
        self.tgt_x_total = tgt_root_data[0]
        self.src_y_total = tgt_root_data[1]

        # ???????????????label
        self.total_label_set = set(self.src_y_total)
        self.pair_label_set = list(self.total_label_set - support_label_set)
        self.pair_label_set.sort()

        # ??????label??????
        self.n_classes = len(self.pair_label_set)

        # ????????????
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

        self.data_shape = self.exp_data.shape

    def __getitem__(self, item):
        exp_data = self.exp_data[item: item + 1]
        exp_label = self.exp_label[item: item + 1]
        exp_label = self.pair_label_set.index(exp_label)

        sim_data = self.sim_data[item: item + 1]
        sim_label = self.sim_label[item: item + 1]
        sim_label = self.pair_label_set.index(sim_label)

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
