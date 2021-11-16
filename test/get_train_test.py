import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import shutil

import os
import math
import numpy as np
import random

from sklearn.utils import shuffle

from config import *
from feature_extraction import *


OBJ_DICT = {
    'top_razor-core-x': '2-1',
    'side_razor-core-x': '2-2',
    'razor-core-x-back': '2-3',  # 新增

    'xiaomi-lamp-top': '3-1',
    'xiaomi-lamp-body': '3-2',
    'xiaomi-lamp-bottom': '3-3',

    'dehumidifier-top': '4-1',
    'dehumidifier-body': '4-2',

    'top2_dell-inspire-case-right-side': '5-1',
    'top1_dell-inspire-case-left-side': '5-2',
    'side_dell-inspire-case-right-side': '5-3',
    'dell-inspire-case-front': '5-4',

    'top_galanz-microwave-oven-body': '6-1',
    'side_galanz-microwave-oven-body': '6-2',
    'galanz-microwave-oven-front': '6-3',

    'white-kettle-top': '7-1',
    'white-kettle-handle': '7-2',
    'white-kettle-body': '7-3',

    'top_philips-speaker-body': '8-1',
    'side_philips-speaker-body': '8-2',

    'top_yamaha-speaker': '9-1',
    'side_yamaha-speaker': '9-2',

    'top_mitsubishi-projector': '10-1',
    'front_mitsubishi-projector': '10-2',

    'hp-printer-top': '12-1',
    'hp-printer-front': '12-2',

    'electic-kettle-top': '14-1',
    'electic-kettle-handle': '14-2',
    'electic-kettle-body': '14-3',

    'top_dual-microwave-oven-side': '15-1',
    'side_dual-microwave-oven-side': '15-2',
    'dual-microwave-oven-front-front': '15-3',

    'side1_hair-dryer': '16-1',
    'side2_hair-dryer': '16-1',
    'side3_hair-dryer': '16-1',

    'weight-meter': '17-1',

    'rice-cooker-top': '18-1',
    'side1_rice-cooker-side': '18-2',
    'side2_rice-cooker-side': '18-3',

    'top_oven-body': '19-1',
    'side_oven-body': '19-2',
    'oven-front': '19-3',
    'oven-panel': '19-4',

    'tv-base': '20-3',

    'coffee-machine-top2': '21-2',
    'coffee-machine-front': '21-3',

    'imac-screen': '22-1',
    'imac-body': '22-2',
}

LABEL_DICT = {
    '2-1': np.int64(1), '2-2': np.int64(2),
    '3-1': np.int64(3), '3-2': np.int64(4), '3-3': np.int64(5),
    '4-1': np.int64(6), '4-2': np.int64(7),
    '5-1': np.int64(8), '5-2': np.int64(9), '5-3': np.int64(10), '5-4': np.int64(46),
    '6-1': np.int64(11), '6-2': np.int64(12), '6-3': np.int64(13),
    '7-1': np.int64(14), '7-2': np.int64(15), '7-3': np.int64(45),
    '8-1': np.int64(16), '8-2': np.int64(17),
    '9-1': np.int64(18), '9-2': np.int64(19),
    '10-1': np.int64(20), '10-2': np.int64(21),
    '12-1': np.int64(22), '12-2': np.int64(23),
    '14-1': np.int64(24), '14-2': np.int64(25), '14-3': np.int64(26),
    '15-1': np.int64(27), '15-2': np.int64(28),
    '16-1': np.int64(29), '16-2': np.int64(30), '16-3': np.int64(31),
    '17-1': np.int64(32),
    '18-1': np.int64(33), '18-2': np.int64(34), '18-3': np.int64(35),
    '19-1': np.int64(36), '19-2': np.int64(37), '19-3': np.int64(38), '19-4': np.int64(39),
    '20-3': np.int64(40),
    '21-2': np.int64(41), '21-3': np.int64(42),
    '22-1': np.int64(43), '22-2': np.int64(44)
}


def get_target_file_name(exp_name, i):
    target_file_name = exp_name + '-' + str(i) + '-1-1.npy'
    return target_file_name


# 将仿真声音文件从源文件夹复制到目标文件夹，并改名
def copy_sim_file(obj_list, overwrite=0):
    root_path = os.path.join(global_var.DATASET, global_var.SOUND_ALL)
    target_path = os.path.join(global_var.DATASET, global_var.RAW_DATA, 'sim_data')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for obj in obj_list.obj_list:
        obj_path = os.path.join(root_path, '%s/mat-%s' % (obj.obj_file_name, obj.material_file_name))
        sound_path = os.path.join(obj_path, obj.sound_prefix + obj.obj_file_name + '_sound' + obj.sound_postfix)
        total_num = np.prod(obj.num)
        for i in range(total_num):
            sound_file_name = obj.sound_prefix + obj.obj_file_name + '-' + str(i + 1) + '.npy'
            sound_file_path = os.path.join(sound_path, sound_file_name)
            target_file_name = get_target_file_name(obj.exp_name, obj.start_num + i)
            target_file_path = os.path.join(target_path, target_file_name)
            if not os.path.exists(target_file_path) or overwrite:
                shutil.copyfile(sound_file_path, target_file_path)


def get_label(file_name):
    surface_id = LABEL_DICT.get('-'.join(file_name.split('-')[:2]), "Not Found!")
    position_id = int(file_name.split('-')[2])
    if surface_id == "Not Found!":
        print(file_name)
        return np.array([0, position_id])
    else:
        return np.array([surface_id, position_id])


# 处理raw_data中的数据
def raw_to_feature_dataset(obj_list, target_path, transform, overwrite=0):
    for path in ['sim_data', 'exp_data']:
        raw_path = os.path.join(global_var.DATASET, global_var.RAW_DATA, path)
        feature_path = os.path.join(global_var.DATASET, global_var.FEATURE_DATA, target_path, path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        for obj in obj_list.obj_list:
            total_num = np.prod(obj.num)
            for i in range(total_num):
                target_file_name = get_target_file_name(obj.exp_name, obj.start_num + i)
                raw_file_path = os.path.join(raw_path, target_file_name)
                feature_file_path = os.path.join(feature_path, target_file_name)
                if not os.path.exists(feature_file_path) or overwrite:
                    raw_file_data = np.load(raw_file_path)
                    if path == 'sim_data':
                        raw_file_data = raw_file_data[0: 1]
                    raw_file_data_transform = transform(raw_file_data)
                    np.save(feature_file_path, raw_file_data_transform)


# 载入feature_data数据
def load_feature_data(obj_list, target_path):
    sim_data, sim_labels = [], []
    exp_data, exp_labels = [], []
    for path in ['sim_data', 'exp_data']:
        feature_path = os.path.join(global_var.DATASET, global_var.FEATURE_DATA, target_path, path)
        for obj in obj_list.obj_list:
            total_num = np.prod(obj.num)
            for i in range(total_num):
                target_file_name = get_target_file_name(obj.exp_name, obj.start_num + i)
                feature_file_path = os.path.join(feature_path, target_file_name)
                feature_file_data = np.load(feature_file_path)
                feature_file_label = get_label(target_file_name)
                if path == 'sim_data':
                    sim_data.append(feature_file_data)
                    sim_labels.append(feature_file_label)
                else:
                    exp_data.append(feature_file_data)
                    exp_labels.append(feature_file_label)

    sim_data, sim_labels = np.array(sim_data), np.array(sim_labels)
    exp_data, exp_labels = np.array(exp_data), np.array(exp_labels)
    sim_data, sim_labels = shuffle(sim_data, sim_labels, random_state=42)
    exp_data, exp_labels = shuffle(exp_data, exp_labels, random_state=42)
    dataset = [sim_data, sim_labels, exp_data, exp_labels]
    
    return dataset


def get_label_set(labels):
    labels_set = set(labels)
    return labels_set.__len__()


# 根据数据类别进行分类，包含所有类别，每类的一部分作为训练集，一部分作为测试集
def get_train_test_dataset_1(dataset, ratio):
    sim_data, sim_label, exp_data, exp_label = dataset
    data_all = np.vstack((sim_data, exp_data))
    labels_all = np.hstack((sim_label, exp_label))
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
        train_data = np.vstack((train_data, data[:num]))
        train_labels = np.hstack((train_labels, labels[:num]))
        test_data = np.vstack((test_data, data[num:]))
        test_labels = np.hstack((test_labels, labels[num:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


# 所有标签都有，但是训练集中只有所有的仿真数据和n个实际敲击，剩下的所有实际敲击都是测试集
def get_train_test_dataset_2(dataset, n):
    sim_data, sim_label, exp_data, exp_label = dataset
    labels_all = np.hstack((sim_label, exp_label))
    labels_set = set(labels_all)

    train_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    train_labels = np.empty((0,), np.int32)
    test_data = np.empty((0, sim_data.shape[1], sim_data.shape[2]), dtype=np.float32)
    test_labels = np.empty((0,), np.int32)

    for label in labels_set:
        data = exp_data[exp_label == label]
        labels = exp_label[exp_label == label]
        train_data = np.vstack((train_data, data[:n]))
        train_labels = np.hstack((train_labels, labels[:n]))
        train_data = np.vstack((train_data, sim_data[sim_label == label]))
        train_labels = np.hstack((train_labels, sim_label[sim_label == label]))
        test_data = np.vstack((test_data, data[n:]))
        test_labels = np.hstack((test_labels, labels[n:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


def get_train_test_dataset_3(dataset, ratio):
    sim_data, sim_label, exp_data, exp_label = dataset
    data_all = np.vstack((sim_data, exp_data))
    labels_all = np.hstack((sim_label, exp_label))
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
        train_data = np.vstack((train_data, data[:num]))
        train_labels = np.hstack((train_labels, labels[:num]))
        test_data = np.vstack((test_data, data[num:]))
        test_labels = np.hstack((test_labels, labels[num:]))

    train_dataset = KnockDataset(train_data, train_labels)
    test_dataset = KnockDataset(test_data, test_labels)

    return train_dataset, test_dataset


class KnockDataset(Dataset):

    def __init__(self, data, labels, transform=None, target_transform=None):
        self.knock_data = data
        self.knock_labels = labels
        self.transform = transform
        self.target_transform = target_transform

        self.label_set = set(self.knock_labels)
        self.n_classes = self.label_set.__len__()

        self.knock_data = torch.Tensor(self.knock_data)
        self.knock_labels = torch.Tensor(self.knock_labels)

        if torch.cuda.is_available():
            self.knock_data = self.knock_data.cuda()
            self.knock_labels = self.knock_labels.cuda()

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
    copy_sim_file(obj_list, 0)
    raw_to_feature_dataset(obj_list, target_path, feature_transform, overwrite)
    dataset = load_feature_data(obj_list, target_path)
    train_dataset_1, test_dataset_1 = get_train_test_dataset_1(dataset, 0.8)
    train_dataset_2, test_dataset_2 = get_train_test_dataset_2(dataset, 5)
