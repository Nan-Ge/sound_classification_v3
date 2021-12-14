import torch
import torch.nn as nn

import numpy as np
import sys
import os
import time

import nni

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from utils.confusion_matrix_plot import plot_confusion_matrix

import config
from trainer_utils import support_mean_vec_generation, cal_euclidean
from plot_val import *


# ------------------------------------------------ 迁移学习训练 ---------------------------------------------------------
def model_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, args, dataset):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')
    err = 0.0
    for epoch in range(0, n_epochs):
        # Train stage
        src_err_lp, tgt_err_lp, err_dc = train_epoch(train_loader=train_loader,
                                                     model=model,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     cuda=cuda,
                                                     epoch=epoch,
                                                     n_epochs=n_epochs,
                                                     args=args)
        scheduler.step()

        # Validation stage: val_loader = (src_val_loader, tgt_val_loader)
        # support_set = (val_loader[1].dataset.val_data, val_loader[1].dataset.val_labels)
        # mean_vec, label_set = support_mean_vec_generation(model, support_set, cuda)  # 生成support set mean vector
        err = test_epoch(val_loader=val_loader,
                             model=model,
                             loss_fn=loss_fn,
                             cuda=cuda)
        # nni.report_intermediate_result(accu)
        # print(', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs))
        print(', validation err of offline training: %.2f for %d / %d' % (err, epoch + 1, n_epochs))

        log_save_name_1 = 'DANN-Triplet-Net_withoutFT_' + exp_time + '.txt'
        with open(os.path.join('../../results/output_train_log_test', log_save_name_1), 'a') as f:
            train_output = '\r epoch: [%d / %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' % (epoch + 1, n_epochs, src_err_lp, tgt_err_lp, err_dc)
            f.write(train_output)
            test_output = ', validation accuracy of offline training: %.2f %% for %d / %d' % (err * 100, epoch + 1, n_epochs)
            f.write(test_output)

        if epoch % 10 == 0:
            # 画图
            plt.close('all')
            for i in range(4):
                embeddings_tl, labels_tl = extract_embeddings(dataset[i], model)
                plot_embeddings(embeddings_tl, embeddings_tl)

    # nni.report_final_result(accu)

    model_name = 'DANN_triplet_baseline_model_' + exp_time + '_' + str(format(err, '.2f')) + '.pkl'
    model_save_path = os.path.join('../../results/output_model_test', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)

    return exp_time


# Pair-wise loss for domain classifier
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs, args):
    model.train()

    src_loader, tgt_loader = train_loader

    batch_src_iter = iter(src_loader)
    batch_tgt_iter = iter(tgt_loader)
    loader_len = len(src_loader) - 1

    src_err_lp, tgt_err_lp, err_dc = 0, 0, 0
    for i in range(loader_len):
        model.zero_grad()
        optimizer.zero_grad()

        (src_data, src_label) = next(batch_src_iter)
        (tgt_data, tgt_label) = next(batch_tgt_iter)
        if type(src_data) not in (tuple, list):
            src_data = (src_data,)
        if type(tgt_data) not in (tuple, list):
            tgt_data = (tgt_data,)

        if type(src_data) in (tuple, list):
            dc_src_label = torch.ones(src_data[0].shape[0]).long()
            dc_tgt_label = torch.ones(src_data[0].shape[0]).long()
        else:
            dc_src_label = torch.ones(src_data.shape[0]).long()
            dc_tgt_label = torch.ones(src_data.shape[0]).long()

        if cuda:
            src_data = tuple(d.cuda() for d in src_data)
            tgt_data = tuple(d.cuda() for d in tgt_data)

            if type(src_label) not in (tuple, list):
                src_label = src_label.cuda()
                tgt_label = tgt_label.cuda()

            dc_src_label = dc_src_label.cuda()
            dc_tgt_label = dc_tgt_label.cuda()

        # (1) Network Forwarding
        src_lp_output, src_dc_output, _ = model(*src_data, eps=config.NOISE_EPS)  # 前向传播
        tgt_lp_output, tgt_dc_output, _ = model(*tgt_data, eps=config.NOISE_EPS)  # 前向传播

        # (2) Loss Function
        # triplet-loss
        src_triple_loss_inputs = (*src_lp_output, src_label)
        src_err_lp = loss_fn[0](*src_lp_output, src_label)

        tgt_triple_loss_inputs = (*tgt_lp_output, tgt_label)
        tgt_err_lp = loss_fn[0](*tgt_lp_output, tgt_label)

        # pair-wise loss
        pair_wise_loss_inputs = (*src_dc_output, tgt_dc_output)
        err_src_dc = loss_fn[1](*src_dc_output, dc_src_label)
        err_tgt_dc = loss_fn[1](*tgt_dc_output, dc_tgt_label)
        err_dc = err_src_dc + err_tgt_dc

        # (3) Total loss & Backward
        total_wgt = args['SRC_TRIP_WGT'] + args['TGT_TRIP_WGT'] + args['PAIR_WGT']
        err_total = args['SRC_TRIP_WGT'] / total_wgt * src_err_lp \
                    + args['TGT_TRIP_WGT'] / total_wgt * tgt_err_lp \
                    + args['PAIR_WGT'] / total_wgt * err_dc
        err_total.backward()
        optimizer.step()

        # (4) Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' %
                         (epoch, i + 1,
                          loader_len,
                          src_err_lp.data.cpu().numpy(),
                          tgt_err_lp.data.cpu().numpy(),
                          err_dc.data.cpu().numpy()))
        sys.stdout.flush()

    return src_err_lp, tgt_err_lp, err_dc


# Validation on the query set
def test_epoch(val_loader, model, loss_fn, cuda):
    n_total = 0
    n_correct = 0  # 浮点型变为整型
    total_err = 0

    with torch.no_grad():
        model.eval()

        src_loader, tgt_loader = val_loader

        batch_src_iter = iter(src_loader)
        batch_tgt_iter = iter(tgt_loader)
        loader_len = len(batch_src_iter) - 1

        for i in range(loader_len):
            (src_data, src_label) = next(batch_src_iter)
            (tgt_data, tgt_label) = next(batch_tgt_iter)
            if type(src_data) not in (tuple, list):
                src_data = (src_data,)
            if type(tgt_data) not in (tuple, list):
                tgt_data = (tgt_data,)

            if cuda:
                src_data = tuple(d.cuda() for d in src_data)
                tgt_data = tuple(d.cuda() for d in tgt_data)

                if type(src_label) not in (tuple, list):
                    src_label = src_label.cuda()
                    tgt_label = tgt_label.cuda()

            # (1) Network Forwarding
            src_lp_output, src_dc_output, _ = model(*src_data, eps=0)  # 前向传播
            tgt_lp_output, tgt_dc_output, _ = model(*tgt_data, eps=0)  # 前向传播

            src_triple_loss_inputs = (*src_lp_output, src_label)
            src_err_lp = loss_fn[0](*src_lp_output, src_label)

            tgt_triple_loss_inputs = (*tgt_lp_output, tgt_label)
            tgt_err_lp = loss_fn[0](*tgt_lp_output, tgt_label)

            total_err = total_err + src_err_lp + tgt_err_lp

        total_err /= loader_len

    return total_err









