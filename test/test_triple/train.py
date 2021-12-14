import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

import config
from dataset import load_data, KnockDataset_pair, KnockDataset_val, BalancedBatchSampler, TripletKnock
from offline_trainer import model_fit
from utils.audio_data_load import arg_list
from utils.feature_extraction import feat_extraction

import nni
from nni.utils import merge_parameter
import argparse
import logging
import json


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

dataset_root_dir = '/mnt/sde/KnockKnock/data/Knock_dataset'
cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

# ///////////////////////////////////////////////// Feature Extraction /////////////////////////////////////////////
feat_data_dir = os.path.join(
    'feature_data', params['FEATURE_TYPE'] + '_' + str(params['INTERVAL']) + '_' + str(params['N_FFT']) + '_' + str(params['DENO_METHOD']))
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
logger.debug("Starting feature extracting!")
feat_extraction(root_data_dir=dataset_root_dir, feat_data_dir=feat_data_dir, kargs=kargs)

# /////////////////////////////////////// Baseline Training & Testing  ///////////////////////////////////////////
feat_dir = os.path.join(dataset_root_dir, feat_data_dir)
dom = ['exp_data', params['SRC_DATASET']]

# (1) 数据集
# 预读取数据
(src_x_total, src_y_total), (tgt_x_total, tgt_y_total) = load_data(dataset_dir=feat_dir, dom=dom, train_flag=1)

'''
3个数据集：
1、pair_wise_dataset：
    除去support_label_set中所有标签后的 / 源域和目标域的 / 成对的 / 80% 样本；
    用于离线阶段 / domain classifier的pair-wise loss；
2、src_val_dataset：
    除去support_label_set中所有标签后的 / 源域的 / 20% 样本；
    用于离线阶段 / 模型的性能验证；
3、tgt_val_dataset：
    除去support_label_set中所有标签后的 / 目标域的 / 20% 样本；
    用于离线阶段 / 模型的性能验证；
'''

# Train
pair_wise_dataset = KnockDataset_pair(
    src_root_data=(src_x_total, src_y_total),
    tgt_root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
batch_sampler = BalancedBatchSampler(
    labels=pair_wise_dataset.exp_label,
    n_classes=pair_wise_dataset.n_classes,
    n_samples=params['NUM_SAMPLES_PER_CLASS'])
train_loader = torch.utils.data.DataLoader(pair_wise_dataset, batch_sampler=batch_sampler)

# Test
src_val_dataset = KnockDataset_val(
    root_data=(src_x_total, src_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
src_val_loader = torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset))

tgt_val_dataset = KnockDataset_val(
    root_data=(tgt_x_total, tgt_y_total),
    support_label_set=config.SUPPORT_SET_LABEL)
tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))

val_loader = (src_val_loader, tgt_val_loader)

# (2) 网络、损失函数、优化器
from losses import OnlineTripletLoss, OnlineTripletLoss_Test, PairWiseLoss, xvec_dann_tripletloss
from loss_utils import SemihardNegativeTripletSelector, KnockPointPairSelector  # Strategies for selecting triplets within a minibatch

# 网络模型
from network import xvec_dann_triplet, xvec_dann_test

# 网络模型
freq_size = pair_wise_dataset.data_shape[1]
seq_len = pair_wise_dataset.data_shape[2]
input_dim = (freq_size, seq_len)

model = xvec_dann_test(input_dim=input_dim, args=params)

if cuda:
    model.cuda()

# 损失函数
# LP_loss = OnlineTripletLoss_Test(params['TRIPLET_MARGIN']/10, SemihardNegativeTripletSelector(params['TRIPLET_MARGIN']/10))  # 分类损失：三元损失
# DC_loss = PairWiseLoss(params['PAIR_MARGIN']/10, KnockPointPairSelector(params['PAIR_MARGIN']/10))  # 域损失：成对损失

margin = 1.
LP_loss = torch.nn.NLLLoss()
DC_loss = torch.nn.NLLLoss()

if cuda:
    LP_loss = LP_loss.cuda()
    DC_loss = DC_loss.cuda()

loss_fn = (LP_loss, DC_loss)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=params['OFF_LR'], weight_decay=params['OFF_WEIGHT_DECAY'])
scheduler = lr_scheduler.StepLR(optimizer, config.OFF_LR_ADJUST_STEP, gamma=config.OFF_LR_ADJUST_RATIO, last_epoch=-1)

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
        args=params
    )
