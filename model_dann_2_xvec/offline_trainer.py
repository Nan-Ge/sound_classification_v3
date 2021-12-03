import torch
from torchsummary import summary

import numpy as np
import sys
import os
import time
import nni
from sklearn.metrics import confusion_matrix

import config
from model_dann_1_xvec.trainer_utils import *


def model_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, args):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    log_save_name_1 = 'model_dann_2_' + exp_time + '.txt'

    with open(os.path.join('../results/output_train_log', log_save_name_1), 'a') as f:
        f.write(str(model))
        f.write("The spectrogram has a size of (%d, %d)" % (train_loader.dataset.data_shape[0], train_loader.dataset.data_shape[1]))

    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')

    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train stage
        src_lp_err, tgt_lp_err, dc_err = train_epoch(
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            cuda=cuda,
            epoch=epoch,
            n_epochs=n_epochs,
            args=args
        )

        # Validation stage [offline_val_loader = (src_val_loader, tgt_val_loader)]
        src_accu_n_correct, src_n_total, _ = val_epoch(
            val_loader=val_loader[0],
            model=model,
            cuda=cuda)

        tgt_accu_n_correct, tgt_n_total, _= val_epoch(
            val_loader=val_loader[1],
            model=model,
            cuda=cuda)

        accu_src = src_accu_n_correct / src_n_total
        accu_tgt = tgt_accu_n_correct / tgt_n_total
        val_accu = (src_accu_n_correct + tgt_accu_n_correct) / (src_n_total + tgt_n_total)

        nni.report_intermediate_result(val_accu)

        print(', validation accuracy of offline training: %.2f %% - %.2f %% for %d / %d' %
              (accu_src * 100, accu_tgt * 100, epoch + 1, n_epochs))

        with open(os.path.join('../results/output_train_log', log_save_name_1), 'a') as f:

            train_output = '\r epoch: [%d / %d], src_lp_err: %f, tgt_lp_err: %f, dc_err: %f' % \
                           (epoch + 1, n_epochs, src_lp_err, tgt_lp_err, dc_err)
            f.write(train_output)

            test_output = ', validation accuracy of offline training: %.2f %% - %.2f %% for %d / %d' % \
                          (accu_src * 100, accu_tgt * 100, epoch + 1, n_epochs)
            f.write(test_output)

    nni.report_final_result(val_accu)

    model_name = 'model_dann_2_' + exp_time + '_' + str(format(accu_src, '.2f')) + '-' + str(format(accu_tgt, '.2f')) + '.pkl'
    model_save_path = os.path.join('../results/output_model', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)

    return exp_time


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs, args):
    model.train()

    batch_iter = iter(train_loader)
    loader_len = len(train_loader) - 1

    for i in range(loader_len):
        p = float(i + epoch * loader_len) / (n_epochs * loader_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.zero_grad()

        (src_data, src_label), (tgt_data, tgt_label) = batch_iter.next()
        src_data = src_data.squeeze()
        src_label = src_label.squeeze()
        tgt_data = tgt_data.squeeze()
        tgt_label = tgt_label.squeeze()

        src_dmn_label = torch.tensor(np.zeros(shape=(src_label.shape[0]), dtype=np.int64))
        tgt_dmn_label = torch.tensor(np.ones(shape=(tgt_label.shape[0]), dtype=np.int64))
        dmn_label = torch.cat((src_dmn_label, tgt_dmn_label))

        optimizer.zero_grad()

        if cuda:
            src_data = src_data.cuda()
            tgt_data = tgt_data.cuda()

            src_label = src_label.cuda()
            tgt_label = tgt_label.cuda()

            dmn_label = dmn_label.cuda()

        # Source Domain 前向传播
        src_feat, src_lp_output, src_dc_output = model(input_data=src_data, alpha=alpha, eps=config.NOISE_EPS)
        # Target Domain 前向传播
        tgt_feat, tgt_lp_output, tgt_dc_output = model(input_data=tgt_data, alpha=alpha, eps=config.NOISE_EPS)

        # (1) Label Predictor Loss
        src_lp_err = loss_fn(src_lp_output, src_label)
        tgt_lp_err = loss_fn(tgt_lp_output, tgt_label)

        # (2) Domain Regressor Loss
        dc_output = torch.cat((src_dc_output, tgt_dc_output), dim=0)
        dc_err = loss_fn(dc_output, dmn_label)

        # (3) Backward Propagation
        err_total = args['SRC_LOSS_WGT'] * src_lp_err + args['TGT_LOSS_WGT'] * tgt_lp_err + args['DC_LOSS_WGT'] * dc_err

        err_total.backward()
        optimizer.step()

        # (4) Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], src_lp_err: %f, tgt_lp_err: %f, dc_err: %f' %
                         (epoch,
                          i + 1,
                          loader_len,
                          src_lp_err.data.cpu().numpy(),
                          tgt_lp_err.data.cpu().numpy(),
                          dc_err.data.cpu().numpy()))
        sys.stdout.flush()

    return src_lp_err, tgt_lp_err, dc_err


def val_epoch(val_loader, model, cuda):
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        model.eval()
        val_iter = iter(val_loader)
        val_sound, val_label = val_iter.next()
        val_sound = val_sound.squeeze()  # 删除channel dimension
        val_label = np.array(val_label.cpu())

        if cuda:
            val_sound = val_sound.cuda()

        # 前向传播
        val_embed, val_pred, _ = model(input_data=val_sound, alpha=0, eps=config.NOISE_EPS)
        pred_label = torch.argmax(val_pred, dim=1)
        pred_label = np.array(pred_label.cpu())

        # 统计准确率
        n_correct += sum(val_label == pred_label)
        n_total += len(val_label)
        accu = n_correct / n_total

        confusion_mat = confusion_matrix(val_label, pred_label)

    return n_correct, n_total, confusion_mat
