import configparser
import numpy as np
import os
import sys
import threading


class GlobalVar:
    _instance_lock = threading.Lock()

    def __init__(self):
        self.ROOT_DIR = '/mnt/sde/wyz/program/ModalSound'
        self.OBJSET_NAME = 'final_obj'
        self.DATASET_NAME = 'sim_result'
        self.SOUND_ALL = 'sound_all'
        self.RAW_DATA = 'raw_data'
        self.FEATURE_DATA = 'feature_data'

        self.MODAL_SOUND = self.ROOT_DIR + '/modal_sound/cmake-build-debug/bin'
        self.ISOSTUFFER = self.MODAL_SOUND + '/isostuffer'
        self.EXTMAT = self.MODAL_SOUND + '/extmat'
        self.MODALEIGEN = self.MODAL_SOUND + '/modal_eigen'
        self.GENMOMENTS = self.MODAL_SOUND + '/gen_moments'
        self.MATLAB = self.ROOT_DIR + '/matlab'
        self.FILEGENERATORS = self.ROOT_DIR + '/file_generator'
        self.DATASET = '/mnt/sde/KnockKnock/data/knock_dataset'
        # self.DATASET = 'E:\Program\GitHub\sound_classification_v3\Knock_dataset'

    def __new__(cls, *args, **kwargs):
        if not hasattr(GlobalVar, "_instance"):
            with GlobalVar._instance_lock:
                if not hasattr(GlobalVar, "_instance"):
                    GlobalVar._instance = object.__new__(cls)
        return GlobalVar._instance


global_var = GlobalVar()


class Obj:

    def __init__(self, obj_name, model_config, material_config):
        self.obj_name = obj_name
        self.obj_file_name = model_config[obj_name]['name']
        self.material_name = model_config[obj_name]['material']

        self.tet = np.array(model_config[obj_name]['tet'].split()).astype(int)
        self.start = np.array(model_config[obj_name]['start'].split()).astype(float)
        self.end = np.array(model_config[obj_name]['end'].split()).astype(float)
        self.num = np.array(model_config[obj_name]['num'].split()).astype(int)
        self.ord = np.array(model_config[obj_name]['ord'].split()).astype(int)
        self.mic = np.array(model_config[obj_name]['mic'].split()).astype(int)
        self.sound_prefix = model_config[obj_name]['sound_prefix']
        if self.sound_prefix:
            self.sound_prefix = self.sound_prefix + '_'
        self.exp_name = model_config[obj_name]['exp_name']
        self.start_num = int(model_config[obj_name]['start_num'])

        self.material_file_name = material_config.get(self.material_name, 'name')
        self.youngs_modulus = material_config.getfloat(self.material_name, 'youngs')
        self.poisson_ratio = material_config.getfloat(self.material_name, 'poisson')
        self.density = material_config.getfloat(self.material_name, 'density')
        self.alpha = material_config.getfloat(self.material_name, 'alpha')
        self.beta = material_config.getfloat(self.material_name, 'beta')
        self.sound_postfix = material_config[self.material_name]['sound_postfix']
        if self.sound_postfix:
            self.sound_postfix = '_' + self.sound_postfix


class ObjList:

    def __init__(self, file_name):
        global_var = GlobalVar()
        model_file_list_name = 'model_' + file_name + '.txt'
        model_file_config_name = 'model_' + file_name + '.cfg'
        material_file_config_name = 'material_' + file_name + '.cfg'

        obj_list_path = os.path.join(global_var.DATASET, 'config', model_file_list_name)
        model_config_path = os.path.join(global_var.DATASET, 'config', model_file_config_name)
        material_config_path = os.path.join(global_var.DATASET, 'config', material_file_config_name)

        self.obj_file_name_list = []
        with open(obj_list_path, encoding='utf-8') as file:
            for line in file:
                obj_name = line.strip()
                if obj_name[0] == '#':
                    continue
                self.obj_file_name_list.append(obj_name)

        model_config = configparser.ConfigParser()
        model_config.read(model_config_path, encoding='utf-8')
        material_config = configparser.ConfigParser()
        material_config.read(material_config_path, encoding='utf-8')

        self.obj_list = []
        for obj_name in self.obj_file_name_list:
            obj = Obj(obj_name, model_config, material_config)
            self.obj_list.append(obj)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class cd:

    def __init__(self, new_path):
        self.newPath = os.path.expanduser(new_path)

    def __enter__(self):
        self.saved_path = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.saved_path)


if __name__ == '__main__':
    obj_list = ObjList('exp')
