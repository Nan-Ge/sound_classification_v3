import torch
import numpy as np
import sys
import os
import time
from sklearn.metrics import confusion_matrix

import model_dann_1_xvec.config as config
from model_dann_1_xvec.offline_trainer import *


def model_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')

    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train stage
        src_err_lp, tgt_err_lp, err_dc = train_epoch(
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            cuda=cuda,
            epoch=epoch,
            n_epochs=n_epochs)

        # Validation stage [offline_val_loader = (src_val_loader, tgt_val_loader)]
        support_set = (val_loader[1].dataset.val_data, val_loader[1].dataset.val_label)
        mean_vec, label_set = support_mean_vec_generation(model, support_set, cuda)  # 生成support set mean vector
        accu, _ = val_epoch(
            val_loader=val_loader[0],
            model=model,
            support_set_mean_vec=mean_vec,
            support_label_set=label_set,
            cuda=cuda)

        print(', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs))

        log_save_name_1 = 'DANN-Triplet-Net_without_FineTuning_' + exp_time + '.txt'
        with open(os.path.join('../results/output_training_log', log_save_name_1), 'a') as f:
            train_output = '\r epoch: [%d / %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' % (epoch + 1, n_epochs, src_err_lp, tgt_err_lp, err_dc)
            f.write(train_output)
            test_output = ', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs)
            f.write(test_output)

    model_name = 'DANN_triplet_baseline_model_' + exp_time + '_' + str(format(accu, '.2f')) + '.pkl'
    model_save_path = os.path.join('../results/output_model', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)

    return exp_time


# Pair-wise loss for domain classifier
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs):
    model.train()

    batch_iter = iter(train_loader)
    loader_len = len(train_loader) - 1

    for i in range(loader_len):
        p = float(i + epoch * loader_len) / (n_epochs * loader_len)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        model.zero_grad()

        (src_data, src_label), (tgt_data, tgt_label) = batch_iter.next()
        src_data = src_data.squeeze()
        tgt_data = tgt_data.squeeze()

        if cuda:
            src_data = src_data.cuda()
            tgt_data = tgt_data.cuda()

        src_feat, tgt_feat, dc_output = model(input_data=(src_data, tgt_data), alpha=alpha)  # 前向传播

        # (1) Triplet Loss
        if type(src_feat) not in (tuple, list):
            src_feat = (src_feat,)
        src_trip_loss_inputs = src_feat
        if src_label is not None:
            src_label = (src_label,)
        src_trip_loss_inputs += src_label
        src_trip_err, _ = loss_fn[0](*src_trip_loss_inputs)

        if type(tgt_feat) not in (tuple, list):
            tgt_feat = (tgt_feat,)
        tgt_trip_loss_inputs = tgt_feat
        if tgt_label is not None:
            tgt_label = (tgt_label,)
        tgt_trip_loss_inputs += tgt_label
        tgt_trip_err, _ = loss_fn[0](*tgt_trip_loss_inputs)

        # (2) Softmax Loss
        dc_err = loss_fn[1](dc_output, src_tgt_label)




        # (4) Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' %
                         (epoch, i + 1,
                          len_dataloader,
                          src_err_lp.data.cpu().numpy(),
                          tgt_err_lp.data.cpu().numpy(),
                          err_dc.data.cpu().numpy()))
        sys.stdout.flush()

    return src_err_lp, tgt_err_lp, err_dc