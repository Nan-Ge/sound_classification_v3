import os
import numpy as np
import glob
import shutil


def get_label_from_filename(filename):
    obj_dict = {
        'razor-core-x': '2-1',
        'razor-core-x-side': '2-2',

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
    return obj_dict[filename]


# 读取原始模拟数据
dataset_path = r'../Knock_dataset/sim_raw_data'
tgt_path = r'../Knock_dataset/raw_data/sim_data'
paths = os.walk(dataset_path)
file_type = 'npy'
for root, dirs, files in paths:
    for file in files:
        if file_type in file:
            obj_name = file.split('-')[0:-1]
            knock_pos = file.split('-')[-1].split('.')[0]
            new_file_name = get_label_from_filename('-'.join(obj_name)) + '-' + knock_pos + '-1-1.npy'
            tgt_file_path = os.path.join(tgt_path, new_file_name)
            shutil.copyfile(os.path.join(root, file), tgt_file_path)

            # temp = np.loadtxt(os.path.join(root, file))
            # np.save(temp, tgt_file_path)

            print(root, file)

