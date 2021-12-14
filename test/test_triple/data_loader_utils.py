import numpy as np
import os
from torch.utils.data.sampler import BatchSampler
from sklearn.utils import shuffle
import shutil

from config import *
from config_2 import *


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
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size


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


# 平均采样
def balanced_sampling(sim_data, exp_data):
    balance_diff = sim_data.shape[0] - exp_data.shape[0]

    if balance_diff > 0:
        rand_slice = np.random.choice(range(0, exp_data.shape[0]), size=balance_diff, replace=True)
        exp_data = np.concatenate((exp_data, exp_data[rand_slice]), axis=0)
    elif balance_diff < 0:
        rand_slice = np.random.choice(range(0, sim_data.shape[0]), size=-balance_diff, replace=True)
        sim_data = np.concatenate((sim_data, sim_data[rand_slice]), axis=0)
    else:
        pass

    return sim_data, exp_data

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
def load_feature_data(obj_list, target_path, balance=False):
    sim_data, sim_labels = None, None
    exp_data, exp_labels = None, None
    feature_file_sim_data, feature_file_sim_label = None, None
    feature_file_exp_data, feature_file_exp_label = None, None
    for obj in obj_list.obj_list:
        total_num = np.prod(obj.num)
        for i in range(total_num):
            for path in ['sim_data', 'exp_data']:
                feature_path = os.path.join(global_var.DATASET, global_var.FEATURE_DATA, target_path, path)
                target_file_name = get_target_file_name(obj.exp_name, obj.start_num + i)
                feature_file_path = os.path.join(feature_path, target_file_name)
                if path == 'sim_data':
                    feature_file_sim_data = np.load(feature_file_path)
                    feature_file_sim_label = get_label(target_file_name)
                else:
                    feature_file_exp_data = np.load(feature_file_path)
                    feature_file_exp_label = get_label(target_file_name)

            if balance:
                feature_file_sim_data, feature_file_exp_data = \
                    balanced_sampling(feature_file_sim_data, feature_file_exp_data)

            if sim_data is None:
                sim_data = feature_file_sim_data
                sim_labels = np.tile(feature_file_sim_label, (feature_file_sim_data.shape[0], 1))
            else:
                sim_data = np.vstack((sim_data, feature_file_sim_data))
                sim_labels = np.vstack(
                    (sim_labels, np.tile(feature_file_sim_label, (feature_file_sim_data.shape[0], 1))))
            if exp_data is None:
                exp_data = feature_file_exp_data
                exp_labels = np.tile(feature_file_exp_label, (feature_file_exp_data.shape[0], 1))
            else:
                exp_data = np.vstack((exp_data, feature_file_exp_data))
                exp_labels = np.vstack(
                    (exp_labels, np.tile(feature_file_exp_label, (feature_file_exp_data.shape[0], 1))))

    sim_data, sim_labels = np.array(sim_data), np.array(sim_labels)
    exp_data, exp_labels = np.array(exp_data), np.array(exp_labels)
    sim_data, sim_labels = shuffle(sim_data, sim_labels, random_state=42)
    exp_data, exp_labels = shuffle(exp_data, exp_labels, random_state=42)
    dataset = [sim_data, sim_labels, exp_data, exp_labels]

    return dataset


def get_label_set(labels):
    labels_set = set(labels)
    return labels_set.__len__()