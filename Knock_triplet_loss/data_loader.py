import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

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


def get_label_object(name):
    if 'changmuban' in name:  # wood
        return np.int64(0)
    elif 'duanmuban' in name:  # wood
        return np.int64(1)
    elif 'lvban' in name:  # lv
        return np.int64(2)
    elif 'tieban' in name:  # tie
        return np.int64(3)
    elif 'xiaozhuodi' in name:  # wood
        return np.int64(4)
    elif 'xiaozhuoding' in name:  # wood
        return np.int64(5)
    elif 'xiaozhuozuo' in name:  # wood
        return np.int64(6)
    elif 'yinxiangbei' in name:  # pvc
        return np.int64(7)
    elif 'yinxiangdi' in name:  # wood
        return np.int64(8)
    elif 'yinxiangding' in name:  # wood
        return np.int64(9)
    elif 'yinxiangyou' in name:  # wood
        return np.int64(10)
    elif 'yklban' in name:  # ykl
        return np.int64(11)
    elif 'zhuogeban' in name:  # wood
        return np.int64(12)


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


def load_data(root_dir, domain):
    path = os.path.join(root_dir, domain)
    x_data_list, y_data_list = [], []
    files = os.listdir(path)
    for i, file in enumerate(files):
        file_name = str(file)
        name, postfix = file_name.split('.')
        if postfix == 'npy':
            np_data = np.load(os.path.join(path, file_name)).astype(np.float32)
            for data in np_data:
                data = data[:, :28, :28]
                data[:, :, 17:28] = 0
                x_data_list.append(data[0])
                y_data_list.append(get_label_object2(name))
    x_data_total = np.array(x_data_list)
    y_data_total = np.array(y_data_list)
    x_data_total = x_data_total[y_data_total != 9]
    y_data_total = y_data_total[y_data_total != 9]
    x_data_total, y_data_total = shuffle(x_data_total, y_data_total)
    return x_data_total, y_data_total


# def get_dataloader(my_x, my_y):
#     tensor_x = torch.Tensor(my_x)
#     tensor_y = torch.Tensor(my_y)
#
#     my_dataset = TensorDataset(tensor_x, tensor_y)
#     my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, drop_last=True)
#     return my_dataloader


class KnockDataset(Dataset):
    def __init__(self, root_dir, domain, train=False, transform=None):
        self.n_classes = 9
        self.train = train
        self.transform = transform
        self.x_data_total, self.y_data_total = load_data(root_dir, domain)
        self.train_data = np.empty((0,
                                    self.x_data_total.shape[1],
                                    self.x_data_total.shape[2]),
                                   dtype=np.float32)
        self.train_labels = np.empty((0,), np.int32)
        self.test_data = np.empty((0,
                                   self.x_data_total.shape[1],
                                   self.x_data_total.shape[2]),
                                  dtype=np.float32)
        self.test_labels = np.empty((0,), np.int32)
        if self.train:
            for i in range(self.n_classes):
                num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
                num_of_data = int(num_of_data * 0.8)
                self.train_data = np.vstack((self.train_data, self.x_data_total[self.y_data_total == i][:num_of_data]))
                self.train_labels = np.hstack((self.train_labels, self.y_data_total[self.y_data_total == i][:num_of_data]))
            self.train_data = torch.Tensor(self.train_data)
            self.train_labels = torch.Tensor(self.train_labels)
        else:
            for i in range(self.n_classes):
                num_of_data = self.y_data_total[self.y_data_total == i].shape[0]
                num_of_data = int(num_of_data * 0.8)
                self.test_data = np.vstack((self.test_data, self.x_data_total[self.y_data_total == i][num_of_data:]))
                self.test_labels = np.hstack((self.test_labels, self.y_data_total[self.y_data_total == i][num_of_data:]))
            self.test_data = torch.Tensor(self.test_data)
            self.test_labels = torch.Tensor(self.test_labels)

    def __getitem__(self, index):
        if self.train:
            img = self.train_data[index: index + 1]
            label = int(self.train_labels[index: index + 1])
        else:
            img = self.test_data[index: index + 1]
            label = int(self.test_labels[index: index + 1])
        return img, label

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


if __name__ == '__main__':
    root_dir = 'E:\\Program\\Acoustic-Expdata-Backend\\Knock_dataset'
    domain = 'exp_data'

    train_dataset = KnockDataset(root_dir, domain, train=True)
    test_dataset = KnockDataset(root_dir, domain, train=False)
    n_classes = train_dataset.n_classes