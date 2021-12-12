import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import shutil
import random
from sklearn.utils import shuffle

from config import *
from config_2 import *
from feature_extraction import *


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
    sim_data, sim_labels = None, None
    exp_data, exp_labels = None, None
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
                    if sim_data is None:
                        sim_data = feature_file_data
                        sim_labels = feature_file_label
                    else:
                        sim_data = np.vstack((sim_data, feature_file_data))
                        sim_labels = np.vstack((sim_labels, feature_file_label))
                else:
                    if exp_data is None:
                        exp_data = feature_file_data
                        exp_labels = np.tile(feature_file_label, (feature_file_data.shape[0], 1))
                    else:
                        exp_data = np.vstack((exp_data, feature_file_data))
                        exp_labels = np.vstack((exp_labels, np.tile(feature_file_label, (feature_file_data.shape[0], 1))))

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


class KnockDataset(Dataset):

    def __init__(self, data, labels, transform=None, target_transform=None, shuff=True):
        self.knock_data = data
        self.knock_labels = labels
        self.transform = transform
        self.target_transform = target_transform

        if shuff:
            self.knock_data, self.knock_labels = shuffle(self.knock_data, self.knock_labels)

        self.label_set = set(self.knock_labels)
        self.n_classes = max(self.label_set)

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
    overwrite = 1
    obj_list = ObjList(name)
    copy_sim_file(obj_list, 1)
    raw_to_feature_dataset(obj_list, target_path, feature_transform, overwrite)
    dataset = load_feature_data(obj_list, target_path)
    # train_dataset_1, test_dataset_1 = get_train_test_dataset_1(dataset, 0.8)
    # train_dataset_2, test_dataset_2 = get_train_test_dataset_2(dataset, 5)
    # train_dataset_3, test_dataset_3 = get_train_test_dataset_3(dataset, 1)
    # train_dataset_4, test_dataset_4 = get_train_test_dataset_4(dataset, 1)
    # train_dataset_5, test_dataset_5 = get_train_test_dataset_5(dataset, 5)
    train_dataset_6, test_dataset_6 = get_train_test_dataset_6(dataset, 5)
