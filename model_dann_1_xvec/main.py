import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from torch.optim import lr_scheduler
import torch.optim as optim
import torch.utils.data

import model_dann_1_xvec.config as config
from model_dann_1_xvec.dataset import load_data, KnockDataset_pair, KnockDataset_val, BalancedBatchSampler
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


def main(args):
    dataset_root_dir = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, 'data', 'Knock_dataset')
    cuda = torch.cuda.is_available()
    torch.cuda.empty_cache()

    # ///////////////////////////////////////////////// Feature Extraction /////////////////////////////////////////////
    feat_data_dir = os.path.join(
        'feature_data', args['FEATURE_TYPE'] + '_' + str(args['INTERVAL']) + '_' + str(args['N_FFT']) + '_' + str(args['DENO_METHOD']))
    kargs = arg_list(
        fs=48000,
        n_fft=args['N_FFT'],
        win_len=args['N_FFT'],
        hop_len=int(args['N_FFT'] / 4),
        n_mels=40,
        window='hann',
        new_len=6192,
        interval=args['INTERVAL'],
        feat_type=args['FEATURE_TYPE'],
        deno_method=args['DENO_METHOD']
    )
    logger.debug("Starting feature extracting!")
    feat_extraction(root_data_dir=dataset_root_dir, feat_data_dir=feat_data_dir, kargs=kargs)

    # /////////////////////////////////////// Baseline Training & Testing  ///////////////////////////////////////////
    feat_dir = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, 'data', 'Knock_dataset', feat_data_dir)
    dom = ['exp_data', args['SRC_DATASET']]

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
        n_samples=args['NUM_SAMPLES_PER_CLASS'])
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
    from model_dann_1_xvec.losses import OnlineTripletLoss, PairWiseLoss
    from model_dann_1_xvec.utils.loss_utils import SemihardNegativeTripletSelector, KnockPointPairSelector  # Strategies for selecting triplets within a minibatch

    # 网络模型
    from network import xvec_dann_triplet

    # 网络模型
    freq_size = pair_wise_dataset.data_shape[1]
    seq_len = pair_wise_dataset.data_shape[2]
    input_dim = (freq_size, seq_len)

    model = xvec_dann_triplet(input_dim=input_dim, args=args)

    if cuda:
        model.cuda()

    # 损失函数
    LP_loss_triplet = OnlineTripletLoss(args['TRIPLET_MARGIN']/10, SemihardNegativeTripletSelector(args['TRIPLET_MARGIN']/10))  # 分类损失：三元损失
    DC_loss_pair = PairWiseLoss(args['PAIR_MARGIN']/10, KnockPointPairSelector(args['PAIR_MARGIN']/10))  # 域损失：成对损失
    loss_fn = (LP_loss_triplet, DC_loss_pair)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args['OFF_LR'], weight_decay=args['OFF_WEIGHT_DECAY'])
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
            args=args
        )

    # # ////////////////////////////////////////////// Fine-tuning & Testing ////////////////////////////////////////////
    # from utils.net_train_utils import model_parameter_printing
    # from model_dann_1_xvec.network import fine_tuned_DANN_triplet_Net
    #
    # # (1) 加载Baseline Model & Baseline Model Test
    # model_path = '../results/output_model'
    # model_list = list(os.listdir(model_path))
    # model_list.sort(reverse=True)
    # model_name = model_list[1]
    # baseline_model = torch.load(os.path.join(model_path, model_name))
    #
    # # (2) 验证基线模型
    # support_set = [(src_train_dataset.train_data, src_train_dataset.train_label),
    #                (src_val_dataset.val_data, src_val_dataset.val_label),
    #                (tgt_train_dataset.train_data, tgt_train_dataset.train_label),
    #                (tgt_val_dataset.val_data, tgt_val_dataset.val_label),
    #                ]
    # query_set = [torch.utils.data.DataLoader(src_train_dataset, batch_size=len(src_train_dataset)),
    #              torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset)),
    #              torch.utils.data.DataLoader(tgt_train_dataset, batch_size=len(tgt_train_dataset)),
    #              torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset)),
    #              ]
    # experiment_index = (3, 1)
    #
    # tgt_sample, src_sample = pair_wise_dataset.__getitem__(random.randint(0, len(pair_wise_dataset) - 1))
    # spec_diff = src_sample - tgt_sample
    #
    # mean_vec, label_set = support_mean_vec_generation(model=baseline_model,
    #                                                   support_set=support_set[experiment_index[0]],
    #                                                   cuda=cuda,
    #                                                   spec_diff=[])
    #
    # accu, cm = val_epoch(query_set[experiment_index[1]], baseline_model, mean_vec, label_set, cuda)
    #
    # plot_confusion_matrix(cm=cm,
    #                       save_path=os.path.join('../results', 'output_confusion_matrix',
    #                                              model_name + '_experiment' + str(experiment_index) + '.png'),
    #                       classes=[str(i) for i in label_set])
    #
    # print('\nBaseline model validation accuracy: %0.2f %%' % accu)
    #
    # # (3) 修改网络结构
    # fine_tuned_model = fine_tuned_DANN_triplet_Net(baseline_model, len(SUPPORT_SET_LABEL))
    #
    # # (4) 固定网络参数
    # fixed_module = ['feature_extractor', 'domain_classifier']
    # # fixed_module = ['domain_classifier']
    # for name, param in fine_tuned_model.named_parameters():
    #     net_module = name.split('.')[0]
    #     if net_module in fixed_module:
    #         param.requires_grad = False
    #
    # model_parameter_printing(fine_tuned_model)  # 打印网络参数
    #
    # # (5) Re-initialize Loss_func and Optimizer
    # loss_fn = torch.nn.NLLLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ON_INITIAL_LR,
    #                        weight_decay=ON_WEIGHT_DECAY)
    # scheduler = lr_scheduler.StepLR(optimizer, ON_LR_ADJUST_STEP, gamma=ON_LR_ADJUST_RATIO, last_epoch=-1)
    #
    # # (6) Fine-tuning & Testing
    # fine_tuning_fit(
    #     train_loader=fine_tuning_set_loader,
    #     test_loader=online_query_loader,
    #     support_label_set=SUPPORT_SET_LABEL,
    #     model=fine_tuned_model,
    #     loss_fn=loss_fn,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     n_epochs=ONLINE_EPOCH,
    #     cuda=cuda
    # )
    #
    # # /////////////////////////////////////////////// 可视化观察 ///////////////////////////////////////////////////////
    # from model_dann_2_xvec.utils.embedding_visualization import tsne_plot
    # tsne_plot(src_train_dataset, tgt_train_dataset, baseline_model, cuda, model_list[0])


if __name__ == '__main__':
    logger = logging.getLogger('XVEC-DANN_AutoML')
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params_from_json('./net_params.json'), tuner_params))
        logger.debug(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise