import torch
import numpy as np
import sys
import os
import time
from sklearn.metrics import confusion_matrix

from model_dann_1_xvec.config import *
from trainer_utils import *


def transfer_baseline_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')
    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train stage
        src_err_lp, tgt_err_lp, err_dc = train_epoch_triplet_pair(train_loader=train_loader,
                                                                  model=model,
                                                                  loss_fn=loss_fn,
                                                                  optimizer=optimizer,
                                                                  cuda=cuda,
                                                                  epoch=epoch,
                                                                  n_epochs=n_epochs)

        # Validation stage [offline_val_loader = (src_val_loader, tgt_val_loader)]
        support_set = (val_loader[1].dataset.val_data, val_loader[1].dataset.val_label)
        mean_vec, label_set = support_mean_vec_generation(model, support_set, cuda)  # 生成support set mean vector
        accu, _ = val_epoch(val_loader=val_loader[0],
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


def train_epoch_triplet_pair(train_loader, model, loss_fn, optimizer, cuda, epoch, n_epochs):
    model.train()

    lp_src_loader = train_loader[0]
    lp_tgt_loader = train_loader[1]
    dc_pair_loader = train_loader[2]

    len_dataloader = max(len(dc_pair_loader), len(lp_src_loader), len(lp_tgt_loader)) - 1

    lp_src_iter = iter(lp_src_loader)
    lp_tgt_iter = iter(lp_tgt_loader)
    dc_pair_iter = iter(dc_pair_loader)

    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / (n_epochs * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # alpha = alpha * 1000

        model.zero_grad()

        # (1) Label Predictor Forwarding
        # 【Source Domain】
        src_data, src_label = lp_src_iter.next()
        src_data = src_data.squeeze()  # 删除channel dimension

        if cuda:
            src_data = src_data.cuda()
            src_label = src_label.cuda()

        # 前向传播
        src_lp_output, _, _ = model(input_data=src_data, alpha=alpha)

        # triplet-loss
        if type(src_lp_output) not in (tuple, list):
            src_lp_output = (src_lp_output,)
        src_triple_loss_inputs = src_lp_output
        if src_label is not None:
            src_label = (src_label,)
            src_triple_loss_inputs += src_label
        src_err_lp, _ = loss_fn[0](*src_triple_loss_inputs)

        # 【Target Domain】
        tgt_data, tgt_label = lp_tgt_iter.next()
        tgt_data = tgt_data.squeeze()  # 删除channel dimension

        if cuda:
            tgt_data = tgt_data.cuda()
            tgt_label = tgt_label.cuda()

        # 前向传播
        tgt_lp_output, _, _ = model(input_data=tgt_data, alpha=alpha)

        # triplet-loss
        if type(tgt_lp_output) not in (tuple, list):
            tgt_lp_output = (tgt_lp_output,)
        tgt_triple_loss_inputs = tgt_lp_output
        if tgt_label is not None:
            tgt_label = (tgt_label,)
            tgt_triple_loss_inputs += tgt_label
        tgt_err_lp, _ = loss_fn[0](*tgt_triple_loss_inputs)

        # (2) Domain Classifier Forwarding
        try:
            data_pair = dc_pair_iter.next()
        except StopIteration:
            dc_pair_iter = iter(dc_pair_loader)
            data_pair = dc_pair_iter.next()

        src_pair_data, tgt_pair_data = data_pair
        src_pair_data = src_pair_data.squeeze()
        tgt_pair_data = tgt_pair_data.squeeze()

        if cuda:
            src_pair_data = src_pair_data.cuda()
            tgt_pair_data = tgt_pair_data.cuda()

        # 前向传播
        _, src_embed_pair, _ = model(input_data=src_pair_data, alpha=alpha)
        _, tgt_embed_pair, _ = model(input_data=tgt_pair_data, alpha=alpha)

        # pair-wise loss
        pair_wise_loss_inputs = (src_embed_pair, tgt_embed_pair)
        err_dc = loss_fn[1](*pair_wise_loss_inputs)

        # (3) Total loss & Backward
        if src_err_lp > 1000:
            src_err_lp, _ = loss_fn[0](*src_triple_loss_inputs)
        elif tgt_err_lp > 1000:
            tgt_err_lp, _ = loss_fn[0](*tgt_triple_loss_inputs)

        err_total = TRIPLET_PAIR_RATIO * (src_err_lp + tgt_err_lp) + err_dc
        err_total.backward()
        optimizer.step()

        # (4) Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], src_err_lp: %f, tgt_err_lp: %f, err_dc: %f' %
                         (epoch, i + 1,
                          len_dataloader,
                          src_err_lp.data.cpu().numpy(),
                          tgt_err_lp.data.cpu().numpy(),
                          err_dc.data.cpu().numpy()))
        sys.stdout.flush()

    return src_err_lp, tgt_err_lp, err_dc


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
        val_embedding, _, _ = model(input_data=val_sound, alpha=0)

        # 相似度计算
        val_embedding = np.array(val_embedding.cpu())
        # cos_sim = cosine_similarity(val_embedding, support_set_mean_vec)
        euc_sim = cal_euclidean(val_embedding, support_set_mean_vec)
        pred_label = np.argmin(euc_sim, axis=1)

        # 将pred_label的值换为统一的label
        for index, label in enumerate(pred_label):
            pred_label[index] = support_label_set[label]

        n_correct += sum(val_label == pred_label)
        n_total += len(val_label)
        accu = n_correct / n_total

        confusion_mat = confusion_matrix(val_label, pred_label)

    return accu * 100, confusion_mat












