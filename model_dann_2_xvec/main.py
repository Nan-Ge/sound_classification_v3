import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.optim import lr_scheduler

import nni
from nni.utils import merge_parameter
import argparse
import logging
import json

from model_dann_1_xvec.dataset import KnockDataset_train, KnockDataset_val, KnockDataset_pair, load_data, BalancedBatchSampler
from offline_trainer import val_epoch, model_fit
from utils.confusion_matrix_plot import plot_confusion_matrix
from utils.audio_data_load import arg_list
from utils.feature_extraction import feat_extraction
import config


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch XVEC-DANN Paras")
    parser.add_argument("--FEATURE_TYPE", type=str,
                        default="stft", help="Method of feature extraction")
    parser.add_argument('--SRC_DATASET', type=str, default="sim_data",
                        help="Training simulation dataset")
    parser.add_argument("--INTERVAL", type=float, default=0.3,
                        help="Interval of feature extraction")
    parser.add_argument("--N_FFT", type=int, default=512,
                        help="FFT window size")
    parser.add_argument('--DENO_METHOD', type=str, default="skimage-Visu",
                        help="Method of exp_data denoising")

    parser.add_argument('--NUM_SAMPLES_PER_CLASS', type=int, default=2,
                        help="Offline batch size")

    parser.add_argument('--EMBED_SIZE', type=int, default=256,
                        help="Feature extractor embed size")
    parser.add_argument('--TDNN_OUT_CHANNEL', type=int, default=512,
                        help="Output channel of TDNN layer")
    parser.add_argument('--TDNN_LAST_OUT_CHANNEL', type=int, default=1024,
                        help="Output channel of the last TDNN layer")
    parser.add_argument('--FC1_OUT_DIM', type=int, default=2,
                        help='Output channel of FC1')
    parser.add_argument('--FC2_OUT_DIM', type=int, default=8,
                        help='Output channel of FC2')
    parser.add_argument('--P_DROP', type=float, default=0.3,
                        help='Probability of dropout')

    parser.add_argument('--OFF_LR', type=float, default=0.0001,
                        help='Learning rate of offline training')
    parser.add_argument('--OFF_WEIGHT_DECAY', type=float, default=0.001,
                        help='Weight decay rate of offline training')

    parser.add_argument('--SRC_LOSS_WGT', type=int, default=15,
                        help='Source Label Predictor Loss')
    parser.add_argument('--TGT_LOSS_WGT', type=int, default=5,
                        help='Target Label Predictor Loss')
    parser.add_argument('--DC_LOSS_WGT', type=int, default=20,
                        help='Domain regressor Loss')

    args, _ = parser.parse_known_args()
    return args


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

    # ///////////////////////////////////////////////// Feature Extraction /////////////////////////////////////////////
    feat_data_dir = os.path.join('feature_data', args['FEATURE_TYPE'] + '_' + str(args['INTERVAL']))
    kargs = arg_list(
        fs=48000,
        n_fft=args['N_FFT'],
        win_len=args['N_FFT'],
        hop_len=int(args['N_FFT'] / 4),
        n_mels=40,
        window='hann',
        max_len=6000,
        interval=args['INTERVAL'],
        feat_type=args['FEATURE_TYPE'],
        deno_method=args['DENO_METHOD']
    )
    logger.debug("Starting feature extracting!")
    feat_extraction(root_data_dir=dataset_root_dir, feat_data_dir=feat_data_dir, kargs=kargs)

    # /////////////////////////////////////// Baseline Training & Testing  ////////////////////////////////////////////
    feat_dir = os.path.join(os.path.pardir, os.path.pardir, os.path.pardir, 'data', 'Knock_dataset', feat_data_dir)
    dom = ['exp_data', args['SRC_DATASET']]
    cuda = torch.cuda.is_available()
    print(cuda)
    torch.cuda.empty_cache()

    # (1) 数据集提取
    # 预读取数据
    (src_x_total, src_y_total), (tgt_x_total, tgt_y_total) = load_data(dataset_dir=feat_dir, dom=dom, train_flag=1)

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
    from network import xvec_dann_orig

    # 网络模型
    freq_size = pair_wise_dataset.data_shape[1]
    seq_len = pair_wise_dataset.data_shape[2]
    input_dim = (freq_size, seq_len)

    model = xvec_dann_orig(
        input_dim=input_dim,
        args=args,
        n_cls=pair_wise_dataset.n_classes,
        version=config.XVEC_VERSION
    )

    if cuda:
        model.cuda()

    # 损失函数
    loss_fn = torch.nn.NLLLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args['OFF_LR'], weight_decay=args['OFF_WEIGHT_DECAY'])
    scheduler = lr_scheduler.StepLR(optimizer, config.OFF_LR_ADJUST_STEP, gamma=config.OFF_LR_ADJUST_RATIO,
                                    last_epoch=-1)

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

    # ////////////////////////////////////////////// Fine-tuning & Testing ////////////////////////////////////////////
    from model_dann_2_xvec.utils.embedding_visualization import tsne_plot
    from model_dann_2_xvec.network import ft_xvec_dann_orig
    from model_dann_1_xvec.dataset import KnockDataset_test
    from utils.net_train_utils import model_parameter_printing
    from online_trainer import netFT_fit

    # (1) 创建Fine-tuning数据集
    # Query Set
    qry_dataset = KnockDataset_test(
        root_data=(src_x_total, src_y_total),
        support_label_set=config.SUPPORT_SET_LABEL)
    qry_loader = torch.utils.data.DataLoader(qry_dataset, batch_size=len(qry_dataset))

    # Support Set
    spt_dataset = KnockDataset_test(
        root_data=(tgt_x_total, tgt_y_total),
        support_label_set=config.SUPPORT_SET_LABEL)
    spt_loader = torch.utils.data.DataLoader(spt_dataset, batch_size=config.FINE_TUNE_BATCH, shuffle=True)

    # (2) 加载Baseline Model
    model_path = '../results/output_model'
    model_list = list(os.listdir(model_path))
    model_list.sort(reverse=True)
    model_name = 'model_dann_2_2021-12-03_20-46-01_0.91-0.92.pkl'
    baseline_model = torch.load(os.path.join(model_path, model_name))

    # (3) 验证基线模型 （混淆矩阵 + t-SNE可视化）
    src_train_dataset = KnockDataset_train(
        root_data=(src_x_total, src_y_total),
        support_label_set=config.SUPPORT_SET_LABEL)
    tgt_train_dataset = KnockDataset_train(
        root_data=(tgt_x_total, tgt_y_total),
        support_label_set=config.SUPPORT_SET_LABEL)

    val_loader_list = [
        torch.utils.data.DataLoader(src_train_dataset, batch_size=len(src_train_dataset)),
        torch.utils.data.DataLoader(src_val_dataset, batch_size=len(src_val_dataset)),
        torch.utils.data.DataLoader(tgt_train_dataset, batch_size=len(tgt_train_dataset)),
        torch.utils.data.DataLoader(tgt_val_dataset, batch_size=len(tgt_val_dataset))]

    n_correct, n_total, cm = val_epoch(val_loader_list[3], baseline_model, cuda)
    accu = n_correct / n_total

    # plot_confusion_matrix(cm=cm,
    #                       save_path=os.path.join('../results', 'output_confusion_matrix',
    #                                              model_name + '_experiment' + '.png'),
    #                       classes=[str(i) for i in pair_wise_dataset.pair_label_set])

    print('\nBaseline model validation accuracy: %0.2f %%' % (accu * 100))

    # tsne_plot(
    #     src_dataset=(src_val_dataset.val_data, src_val_dataset.val_label),
    #     tgt_dataset=(tgt_val_dataset.val_data, tgt_val_dataset.val_label),
    #     model=baseline_model,
    #     model_name=model_name,
    #     cuda=cuda)

    # (4) 定义Fine-tuning网络及可训练参数
    ft_model = ft_xvec_dann_orig(
        baseModel=baseline_model,
        args=args,
        n_class=len(config.SUPPORT_SET_LABEL),
        version=config.XVEC_VERSION)

    for name, param in ft_model.named_parameters():
        net_module = name.split('.')[0]
        if net_module in config.FIXED_MODULE:
            param.requires_grad = False
        # if name == 'feature_extractor.fc1.weight' or name == 'feature_extractor.fc1.bias':
        #     param.requires_grad = True

    model_parameter_printing(ft_model)  # 打印网络参数

    # (5) Re-initialize Loss_func and Optimizer
    loss_fn = torch.nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ft_model.parameters()), lr=config.ON_INITIAL_LR,
                           weight_decay=config.ON_WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, config.ON_LR_ADJUST_STEP, gamma=config.ON_LR_ADJUST_RATIO, last_epoch=-1)

    # (6) Fine-tuning & Testing
    if config.FINE_TUNE_STAGE:
        netFT_fit(
            train_loader=spt_loader,
            test_loader=qry_loader,
            support_label_set=config.SUPPORT_SET_LABEL,
            model=ft_model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=config.ONLINE_EPOCH,
            cuda=cuda
        )


if __name__ == '__main__':
    logger = logging.getLogger('XVEC-DANN_AutoML')
    try:
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params_from_json('./net_params.json'), tuner_params))
        logger.debug(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        print(exception)
        raise
