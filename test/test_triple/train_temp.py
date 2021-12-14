import os
import sys

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

import config
from data_loader import *
from offline_trainer_temp import model_fit
from utils.audio_data_load import arg_list
from utils.feature_extraction import feat_extraction

import nni
from nni.utils import merge_parameter
import argparse
import logging
import json

from dataset_temp import TripletKnock


def get_params_from_json(json_file_path):
    parser = argparse.ArgumentParser(description="PyTorch XVEC-DANN Paras")
    args_dict = vars(parser.parse_args())

    with open(json_file_path) as f:
        params_dict = json.load(fp=f)

    for key in params_dict.keys():
        args_dict[key] = params_dict[key]

    return argparse.Namespace(**args_dict)


logger = logging.getLogger('XVEC-DANN_AutoML')
# tuner_params = nni.get_next_parameter()
params = vars(get_params_from_json('./net_params.json'))

cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

# ///////////////////////////////////////////////// Feature Extraction /////////////////////////////////////////////
kargs = arg_list(
    fs=48000,
    n_fft=params['N_FFT'],
    win_len=params['N_FFT'],
    hop_len=int(params['N_FFT'] / 4),
    n_mels=40,
    window='hann',
    new_len=6192,
    interval=params['INTERVAL'],
    feat_type=params['FEATURE_TYPE'],
    deno_method=params['DENO_METHOD']
)

# /////////////////////////////////////// Baseline Training & Testing  ///////////////////////////////////////////
# (1) 数据集
# 预读取数据
feat_data_dir = os.path.join(
    params['FEATURE_TYPE'] + '_' + str(params['INTERVAL']) + '_' + str(params['N_FFT']) + '_' + str(params['DENO_METHOD']))
logger.debug("Starting feature extracting!")
feat_extraction(root_data_dir=global_var.DATASET, feat_data_dir=os.path.join(global_var.FEATURE_DATA, feat_data_dir), kargs=kargs)

name = 'exp'
target_path = feat_data_dir
# feature_transform = fbank_transform
overwrite = 0
obj_list = ObjList(name)
# copy_sim_file(obj_list, 1)
# raw_to_feature_dataset(obj_list, target_path, feature_transform, overwrite)
dataset = load_feature_data(obj_list, feat_data_dir, balance=True)
train_src_dataset, train_tgt_dataset, test_src_dataset, test_tgt_dataset = get_train_test_dataset_src_tgt_1(dataset, 0.8)
batch_size = int(train_src_dataset.n_classes * params['NUM_SAMPLES_PER_CLASS'])

# Triplet
triplet_train_src_dataset = TripletKnock(train_src_dataset)
triplet_train_tgt_dataset = TripletKnock(train_tgt_dataset)
triplet_test_src_dataset = TripletKnock(test_src_dataset)
triplet_test_tgt_dataset = TripletKnock(test_tgt_dataset)

# Train
train_src_loader = torch.utils.data.DataLoader(triplet_train_src_dataset, batch_size=batch_size, shuffle=True)
train_tgt_loader = torch.utils.data.DataLoader(triplet_train_tgt_dataset, batch_size=batch_size, shuffle=True)
train_loader = (train_src_loader, train_tgt_loader)

# Test
val_src_loader = torch.utils.data.DataLoader(triplet_test_src_dataset, batch_size=batch_size, shuffle=True)
val_tgt_loader = torch.utils.data.DataLoader(triplet_test_tgt_dataset, batch_size=batch_size, shuffle=True)
val_loader = (val_src_loader, val_tgt_loader)

# (2) 网络、损失函数、优化器
from losses import OnlineTripletLoss, OnlineTripletLoss_Test, PairWiseLoss, xvec_dann_tripletloss, xvec_dann_dc_tripletloss
from loss_utils import SemihardNegativeTripletSelector, KnockPointPairSelector  # Strategies for selecting triplets within a minibatch

# 网络模型
from network import xvec_dann_triplet, xvec_dann_test, xvec_dann_triplet_test

# 网络模型
freq_size = train_src_dataset.data_shape[1]
seq_len = train_src_dataset.data_shape[2]
input_dim = (freq_size, seq_len)

model = xvec_dann_triplet_test(input_dim=input_dim, args=params)

if cuda:
    model.cuda()

# 损失函数
# LP_loss = OnlineTripletLoss_Test(params['TRIPLET_MARGIN']/10, SemihardNegativeTripletSelector(params['TRIPLET_MARGIN']/10))  # 分类损失：三元损失
# DC_loss = PairWiseLoss(params['PAIR_MARGIN']/10, KnockPointPairSelector(params['PAIR_MARGIN']/10))  # 域损失：成对损失

margin = 1.
LP_loss = xvec_dann_tripletloss(margin)
DC_loss = xvec_dann_dc_tripletloss()

if cuda:
    LP_loss = LP_loss.cuda()
    DC_loss = DC_loss.cuda()

loss_fn = (LP_loss, DC_loss)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=params['OFF_LR'], weight_decay=params['OFF_WEIGHT_DECAY'])
scheduler = lr_scheduler.StepLR(optimizer, config.OFF_LR_ADJUST_STEP, gamma=config.OFF_LR_ADJUST_RATIO, last_epoch=-1)

dataset_temp = (train_src_dataset, train_tgt_dataset, test_src_dataset, test_tgt_dataset)
# (3) Baseline model Training & Testing
if config.TRAIN_STAGE:
    model_fit(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=config.OFFLINE_EPOCH,
        cuda=cuda,
        args=params,
        dataset=dataset_temp
    )
