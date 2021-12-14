import torch
from torch.utils.data import Dataset
import random
import sklearn.utils

from config import *
from config_2 import *
from feature_extraction import *
from data_loader_utils import *


# 根据数据类别进行分类，包含所有类别，每类的一部分作为训练集，一部分作为测试集
def get_train_test_dataset_1(dataset, ratio):
    sim_data, sim_label, exp_data, exp_label = dataset
    data_all = np.vstack((sim_data, exp_data))
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        index = labels_all == label
        num = int(np.sum(index) * ratio)
        data = data_all[index]
        labels = labels_all[index]
        train_data = np.vstack((train_data, data[0: num]))
        train_labels = np.hstack((train_labels, labels[0: num]))
        test_data = np.vstack((test_data, data[num:]))
        test_labels = np.hstack((test_labels, labels[num:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中只有所有的仿真数据和n个实际敲击，剩下的所有实际敲击都是测试集
def get_train_test_dataset_2(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        data = exp_data[exp_label[:, 0] == label]
        labels = exp_label[:, 0][exp_label[:, 0] == label]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        train_data = np.vstack((train_data, sim_data[sim_label[:, 0] == label]))
        train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        test_data = np.vstack((test_data, data[n:]))
        test_labels = np.hstack((test_labels, labels[n:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中只有n个实际敲击，剩下的所有实际敲击都是测试集
def get_train_test_dataset_3(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        data = exp_data[exp_label[:, 0] == label]
        labels = exp_label[:, 0][exp_label[:, 0] == label]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        # train_data = np.vstack((train_data, sim_data[sim_label[:, 0] == label]))
        # train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        test_data = np.vstack((test_data, data[n:]))
        test_labels = np.hstack((test_labels, labels[n:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中只有所有的仿真数据和n个实际敲击，剩下的所有实际敲击都是测试集，使用敲击样例利用残差矫正仿真数据
def get_train_test_dataset_4(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        data = exp_data[exp_label[:, 0] == label]
        labels = exp_label[:, 0][exp_label[:, 0] == label]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        sim_data_transform = sim_data[sim_label[:, 0] == label]
        diff = np.average(train_data, axis=0) - np.average(sim_data_transform, axis=0)
        sim_data_transform = sim_data_transform + diff
        train_data = np.vstack((train_data, sim_data_transform))
        train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        test_data = np.vstack((test_data, data[n:]))
        test_labels = np.hstack((test_labels, labels[n:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中每个面只有某个点的敲击数据，剩下的敲击数据是训练集
def get_train_test_dataset_5(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        surface_label = exp_label[exp_label[:, 0] == label][:, 1]
        surface_id = random.choice(surface_label)
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        # train_data = np.vstack((train_data, sim_data[sim_label[:, 0] == label]))
        # train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        if data.shape[0] < n:
            test_data = np.vstack((test_data, data[n:]))
            test_labels = np.hstack((test_labels, labels[n:]))
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        test_data = np.vstack((test_data, data))
        test_labels = np.hstack((test_labels, labels))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中每个面有某个点的敲击数据和所有的仿真数据，剩下的敲击数据是训练集
def get_train_test_dataset_6(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        surface_label = exp_label[exp_label[:, 0] == label][:, 1]
        surface_id = random.choice(surface_label)
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        train_data = np.vstack((train_data, sim_data[sim_label[:, 0] == label]))
        train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        if data.shape[0] < n:
            test_data = np.vstack((test_data, data[n:]))
            test_labels = np.hstack((test_labels, labels[n:]))
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        test_data = np.vstack((test_data, data))
        test_labels = np.hstack((test_labels, labels))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中每个面有某个点的敲击数据和所有的仿真数据，剩下的敲击数据是训练集，不带迁移
def get_train_test_dataset_7(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        surface_label = exp_label[exp_label[:, 0] == label][:, 1]
        surface_id = random.choice(surface_label)
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] == surface_id)]
        train_data = np.vstack((train_data, data[0: n]))
        train_labels = np.hstack((train_labels, labels[0: n]))
        train_data = np.vstack((train_data, sim_data[sim_label[:, 0] == label]))
        train_labels = np.hstack((train_labels, sim_label[:, 0][sim_label[:, 0] == label]))
        if data.shape[0] < n:
            test_data = np.vstack((test_data, data[n:]))
            test_labels = np.hstack((test_labels, labels[n:]))
        data = exp_data[np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        labels = exp_label[:, 0][np.logical_and(exp_label[:, 0] == label, exp_label[:, 1] != surface_id)]
        test_data = np.vstack((test_data, data))
        test_labels = np.hstack((test_labels, labels))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


def get_train_test_dataset_src_tgt_1(dataset, ratio):
    sim_data, sim_label, exp_data, exp_label = dataset
    data_all = np.vstack((sim_data, exp_data))
    labels_all = np.hstack((sim_label[:, 0], exp_label[:, 0]))
    labels_set = set(labels_all)

    train_src_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_tgt_data = np.empty((0, exp_data.shape[1], exp_data.shape[2]), dtype=np.float32)
    train_src_labels = np.empty((0,), np.int32)
    train_tgt_labels = np.empty((0,), np.int32)
    test_src_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_tgt_data = np.empty((0, exp_data.shape[1], exp_data.shape[2]), dtype=np.float32)
    test_src_labels = np.empty((0,), np.int32)
    test_tgt_labels = np.empty((0,), np.int32)

    for label in labels_set:
        index = labels_all == label
        num = int(np.sum(index) * ratio)
        data = data_all[index]
        labels = labels_all[index]
        train_src_data = np.vstack((train_src_data, data[0: num]))
        train_src_labels = np.hstack((train_src_labels, labels[0: num]))
        train_tgt_data = np.vstack((train_tgt_data, data[0: num]))
        train_tgt_labels = np.hstack((train_tgt_labels, labels[0: num]))
        test_src_data = np.vstack((test_src_data, data[num:]))
        test_src_labels = np.hstack((test_src_labels, labels[num:]))
        test_tgt_data = np.vstack((test_tgt_data, data[num:]))
        test_tgt_labels = np.hstack((test_tgt_labels, labels[num:]))

    train_src_dataset = KnockDataset(train_src_data, train_src_labels)
    train_tgt_dataset = KnockDataset(train_tgt_data, train_tgt_labels)
    test_src_dataset = KnockDataset(test_src_data, test_src_labels, train=False)
    test_tgt_dataset = KnockDataset(test_tgt_data, test_tgt_labels, train=False)

    # return train_src_data, train_src_labels, train_tgt_data, train_tgt_labels, \
    #        test_src_data, test_src_labels, test_tgt_data, test_tgt_labels
    return train_src_dataset, train_tgt_dataset, test_src_dataset, test_tgt_dataset



class KnockDataset(Dataset):

    def __init__(self, data, labels, transform=None, target_transform=None, train=True, shuffle=True):
        self.knock_data = data
        self.knock_labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.data_shape = self.knock_data.shape

        if shuffle:
            self.knock_data, self.knock_labels = sklearn.utils.shuffle(self.knock_data, self.knock_labels)

        self.label_set = set(self.knock_labels)
        label = list(set(self.label_set))
        for i in range(label.__len__()):
            self.knock_labels[self.knock_labels == label[i]] = i
        self.n_classes = label.__len__()

        self.knock_data = torch.Tensor(self.knock_data)
        self.knock_labels = torch.Tensor(self.knock_labels)

        # if torch.cuda.is_available():
        #     self.knock_data = self.knock_data.cuda()
        #     self.knock_labels = self.knock_labels.cuda()

    def __getitem__(self, index):
        knock_data = self.knock_data[index]
        knock_label = int(self.knock_labels[index])
        if self.transform:
            knock_data = self.transform(knock_data)
        if self.target_transform:
            knock_label = self.target_transform(knock_label)
        return knock_data, knock_label

    def __len__(self):
        return len(self.knock_labels)


if __name__ == '__main__':
    name = 'exp'
    target_path = 'fbank_dnoised_data'
    feature_transform = fbank_transform
    overwrite = 0
    obj_list = ObjList(name)
    copy_sim_file(obj_list, 1)
    raw_to_feature_dataset(obj_list, target_path, feature_transform, overwrite)
    dataset = load_feature_data(obj_list, target_path, balance=True)
    # train_dataset_1, test_dataset_1 = get_train_test_dataset_1(dataset, 0.8)
    # train_dataset_2, test_dataset_2 = get_train_test_dataset_2(dataset, 5)
    # train_dataset_3, test_dataset_3 = get_train_test_dataset_3(dataset, 1)
    # train_dataset_4, test_dataset_4 = get_train_test_dataset_4(dataset, 1)
    # train_dataset_5, test_dataset_5 = get_train_test_dataset_5(dataset, 5)
    train_src_dataset, train_tgt_dataset, test_src_dataset, test_tgt_dataset = get_train_test_dataset_src_tgt_1(dataset, 0.8)
