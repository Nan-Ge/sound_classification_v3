import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from config import *
import shutil

import os
import math
import numpy as np
import random

from sklearn.utils import shuffle

OBJ_DICT = {
    'top_razor-core-x': '2-1',
    'side_razor-core-x': '2-2',
    'razor-core-x-back': '2-3',  # 新增

    'xiaomi-lamp-top': '3-1',
    'xiaomi-lamp-body': '3-2',
    'xiaomi-lamp-bottom': '3-3',

    'dehumidifier-top': '4-1',
    'dehumidifier-body': '4-2',

    'top2_dell-inspire-case-right-side': '  ',
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
    'side2_hair-dryer': '16-2',
    'side3_hair-dryer': '16-3',

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


def get_target_file_name(sound_file_name, i):
    target_file_name = OBJ_DICT[sound_file_name] + str(i) + '-1-1.npy'
    return target_file_name


# 将声音文件从源文件夹复制到目标文件夹，并改名
def copy_file(obj_list):
    root_path = os.path.join(global_var.DATASET, global_var.SOUND_ALL)
    target_path = os.path.join(global_var.DATASET, global_var.RAW_DATA, 'sim_data')
    for obj in obj_list.obj_list:
        obj_path = os.path.join(root_path, '%s/mat-%s' % (obj.obj_file_name, obj.material_file_name))
        sound_path = os.path.join(obj_path, obj.sound_prefix + obj.obj_file_name + '_sound' + obj.sound_postfix)
        total_num = np.prod(obj.num)
        for i in range(total_num):
            sound_file_name = obj.sound_prefix + obj.obj_file_name + '-' + str(obj.start_num + i) + '.npy'
            sound_file_path = os.path.join(sound_path, sound_file_name)
            target_file_name = get_target_file_name(obj.sound_prefix + obj.obj_file_name, obj.start_num + i)
            target_file_path = os.path.join(target_path, target_file_name)
            shutil.copyfile(sound_file_path, target_file_path)


def get_label(file_name):
    res = LABEL_DICT.get('-'.join(file_name.split('-')[0:2]), "Not Found!")
    if res == "Not Found!":
        print(file_name)
        return np.int64(0)
    else:
        return res


# 将所有声音npy数据读取到内存
def get_dataset(obj_list):
    # 读取仿真数据
    sim_data = []
    sim_label = []
    sim_path = os.path.join(global_var.DATASET, global_var.RAW_DATA, 'sim_data')
    for obj in obj_list.obj_list:
        total_num = np.prod(obj.num)
        for i in range(total_num):
            sim_file_name = get_target_file_name(obj.sound_prefix + obj.obj_file_name, obj.start_num + i)
            sim_file_path = os.path.join(sim_path, sim_file_name)
            sim_file_data = np.load(sim_file_path)
            sim_data.append(sim_file_data)
            label = get_label(sim_file_name)
            sim_label.append(label)

    # 读取实验数据
    exp_data = []
    exp_label = []
    exp_path = os.path.join(global_var.DATASET, global_var.RAW_DATA, 'exp_data')
    for obj in obj_list.obj_list:
        total_num = np.prod(obj.num)
        for i in range(total_num):
            exp_file_name = get_target_file_name(obj.sound_prefix + obj.obj_file_name, obj.start_num + i)
            exp_file_path = os.path.join(exp_path, exp_file_name)
            exp_file_data = np.load(exp_file_path)
            exp_data.append(exp_file_data)
            label = get_label(exp_file_name)
            exp_label.append(label)

    dataset = [sim_data, sim_label, exp_data, exp_label]
    
    return dataset


def get_train_test_dataset(dataset):
    train_dataset, test_dataset = 0, 0
    return train_dataset, test_dataset


if __name__ == '__main__':
    name = 'exp'
    obj_list = ObjList(name)
    copy_file(obj_list)
    dataset = get_dataset(obj_list)
    train_dataset, test_dataset = get_train_test_dataset(dataset)
