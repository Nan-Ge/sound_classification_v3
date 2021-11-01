import os
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

        'white-kettle-top': '7-1',
        'white-kettle-handle': '7-2',
        'white-kettle-body': '7-3',
        'philips-speaker-body': '8-2',
        'philips-speaker-back': '8-3',

        'yamaha-speaker': '9-2',
        'top_mitsubishi-projector': '10-1',
        'front_mitsubishi-projector': '10-2',

        'weight-meter': '17-1',
        'rice-cooker-top': '18-1',
        'side1_rice-cooker-side': '18-2',
        'side2_rice-cooker-side': '18-3',

        'tv-base': '20-3',
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
            print(root, file)

