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

import model_dann_1_xvec.config as config
from model_dann_1_xvec.utils.trainer_utils import support_mean_vec_generation, cal_euclidean


# ------------------------------------------------ 迁移学习训练 ---------------------------------------------------------
def model_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, args):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')
    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train stage
        src_err_lp, tgt_err_lp, err_dc = train_epoch(train_loader=train_loader,
                                                     model=model,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     cuda=cuda,
                                                     epoch=epoch,
                                                     n_epochs=n_epochs,
                                                     args=args)

        # Validation stage: val_loader = (src_val_loader, tgt_val_loader)
        support_set = (val_loader[1].dataset.val_data, val_loader[1].dataset.val_label)
        mean_vec, label_set = support_mean_vec_generation(model, support_set, cuda)  # 生成support set mean vector
        accu, _ = val_epoch(val_loader=val_loader[0],
                            model=model,
                            support_set_mean_vec=mean_vec,
                            support_label_set=label_set,
                            cuda=cuda)
        nni.report_intermediate_result(accu)
        print(', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs))

        log_save_name_1 = 'DANN-Triplet-Net_withoutFT_' + exp_time + '.txt'
        with open(os.path.join('../results/output_train_log', log_save_name_1), 'a') as f:
            train_output = '\r epoch: [%d / %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' % (epoch + 1, n_epochs, src_err_lp, tgt_err_lp, err_dc)
            f.write(train_output)
            test_output = ', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs)
            f.write(test_output)

    nni.report_final_result(accu)

    model_name = 'DANN_triplet_baseline_model_' + exp_time + '_' + str(format(accu, '.2f')) + '.pkl'
    model_save_path = os.path.join('../results/output_model', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)

    return exp_time


# Pair-wise loss for domain classifier
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs, args):
    model.train()

    batch_iter = iter(train_loader)
    loader_len = len(train_loader) - 1

    for i in range(loader_len):
        model.zero_grad()
        optimizer.zero_grad()

        (src_data, src_label), (tgt_data, tgt_label) = batch_iter.next()
        src_data = src_data.squeeze()
        src_label = src_label.squeeze()
        tgt_data = tgt_data.squeeze()
        tgt_label = tgt_label.squeeze()

        if cuda:
            src_data = src_data.cuda()
            tgt_data = tgt_data.cuda()

            src_label = src_label.cuda()
            tgt_label = tgt_label.cuda()

        # (1) Network Forwarding
        src_lp_output, src_embed_pair, _ = model(input_data=src_data, eps=config.NOISE_EPS)  # 前向传播
        tgt_lp_output, tgt_embed_pair, _ = model(input_data=tgt_data, eps=config.NOISE_EPS)  # 前向传播

        # (2) Loss Function
        # triplet-loss
        src_triple_loss_inputs = (src_lp_output, src_label)
        src_err_lp, _ = loss_fn[0](*src_triple_loss_inputs)

        tgt_triple_loss_inputs = (tgt_lp_output, tgt_label)
        tgt_err_lp, _ = loss_fn[0](*tgt_triple_loss_inputs)

        # pair-wise loss
        pair_wise_loss_inputs = (src_embed_pair, tgt_embed_pair)
        err_dc = loss_fn[1](*pair_wise_loss_inputs)

        # (3) Total loss & Backward
        total_wgt = args['SRC_TRIP_WGT'] + args['TGT_TRIP_WGT'] + args['PAIR_WGT']
        err_total = args['SRC_TRIP_WGT'] / total_wgt * src_err_lp + args['TGT_TRIP_WGT'] / total_wgt * tgt_err_lp + args['PAIR_WGT'] / total_wgt * err_dc
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


# Overall test on the query set
def test_epoch(test_loader, model, support_set_mean_vec, support_label_set, cuda):
    n_total = 0
    n_correct = 0
    support_label_set = [int(i) for i in iter(support_label_set)]  # 浮点型变为整型

    with torch.no_grad():
        model.eval()
        test_iter = iter(test_loader)
        test_sound, test_label = test_iter.next()
        test_sound = test_sound.squeeze()  # 删除channel dimension
        test_label = np.array(test_label.cpu())

        if cuda:
            test_sound = test_sound.cuda()

        test_embedding, _, _ = model(input_data=test_sound, alpha=0)
        test_embedding = np.array(test_embedding.cpu())
        cos_sim = cosine_similarity(test_embedding, support_set_mean_vec)
        pred_label = np.argmax(cos_sim, axis=1)

        # 将pred_label的值换为统一的label
        for index, label in enumerate(pred_label):
            pred_label[index] = support_label_set[label]

        n_correct += sum(test_label == pred_label)
        n_total += len(test_label)
        accu = n_correct / n_total

    return accu


# Validation on the query set
def val_epoch(val_loader, model, support_set_mean_vec, support_label_set, cuda):
    n_total = 0
    n_correct = 0
    support_label_set = [int(i) for i in iter(support_label_set)]  # 浮点型变为整型

    with torch.no_grad():
        model.eval()
        val_iter = iter(val_loader)
        val_sound, val_label = val_iter.next()
        val_sound = val_sound.squeeze()  # 删除channel dimension
        val_label = np.array(val_label.cpu())

        if cuda:
            val_sound = val_sound.cuda()

        # 前向传播
        val_embedding, _, _ = model(input_data=val_sound, eps=0)

        # 相似度计算
        val_embedding = np.array(val_embedding.cpu())
        # cos_sim = cosine_similarity(val_embedding, support_set_mean_vec)
        euc_sim = cal_euclidean(val_embedding, support_set_mean_vec)
        pred_label = np.argmin(euc_sim, axis=1)

        # 统计
        n_correct += sum(val_label == pred_label)
        n_total += len(val_label)
        accu = n_correct / n_total

        confusion_mat = confusion_matrix(val_label, pred_label)

    return accu, confusion_mat









