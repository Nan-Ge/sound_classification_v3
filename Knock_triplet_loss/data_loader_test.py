import torch
from torch.utils.data import TensorDataset, DataLoader

import os
import numpy as np


def get_label(name):
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
    return np.int(0)


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
                x_data_list.append(data)
                y_data_list.append(get_label(name))
    x_data_total = np.array(x_data_list)
    y_data_total = np.array(y_data_list)
    return x_data_total, y_data_total


def get_dataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True, drop_last=True)
    return my_dataloader


if __name__ == '__main__':
    pass
