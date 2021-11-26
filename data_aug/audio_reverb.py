import os
import sys
import shutil
import numpy as np

from pysndfx import AudioEffectsChain
from librosa import load

from utils.feature_extraction import load_npy


if __name__ == '__main__':

    max_len = 6000
    interval = [0, 1]

    fx = (
        AudioEffectsChain()
        .reverb(
            reverberance=50,
            room_scale=10,
            stereo_depth=100,
            hf_damping=0,
            pre_delay=0,
            wet_gain=0,
            wet_only=False
        )
    )

    sim_data_aug_dir = '../Knock_dataset/raw_data/sim_data_aug'
    sim_data_dir = '../Knock_dataset/raw_data/sim_data'

    shutil.rmtree(sim_data_aug_dir)
    os.mkdir(sim_data_aug_dir)

    sim_audio_files = os.listdir(sim_data_dir)

    for index, audio_file in enumerate(sim_audio_files):


        # inFile = os.path.join(sim_data_dir, audio_file)
        # outFile = os.path.join(sim_data_aug_dir, audio_file.replace('.wav', '.ogg'))
        # fx(inFile, outFile)

        audio_arr = load_npy(audio_filepath=os.path.join(sim_data_dir, audio_file),
                             max_len=max_len,
                             interval=interval)

        # ---- 只处理1个样本 ----
        aug_audio_arr = fx(audio_arr[0])[np.newaxis, :]

        # ---- 处理所有样本 ----
        # aug_audio_arr = np.empty(shape=(0, audio_arr.shape[1]), dtype=audio_arr.dtype)
        # for data in audio_arr:
        #     aug_data = fx(data)[np.newaxis, :]
        #     aug_audio_arr = np.concatenate((aug_audio_arr, aug_data), axis=0)

        np.save(os.path.join(sim_data_aug_dir, audio_file), aug_audio_arr)
        sys.stdout.write('\r Current file %d / %d' % (index + 1, len(sim_audio_files)))







