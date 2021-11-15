import torch
import torch.nn as nn
from config import *

import numpy as np
import sys
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from utils.confusion_matrix_plot import plot_confusion_matrix


# --------------------------------------------- 非迁移学习训练 -----------------------------------------------------------
def non_transfer_base_line_fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    print('\n--------- Start transfer baseline model training at:' + exp_time + '---------')
    for epoch in range(0, n_epochs):
        scheduler.step()
        # Train Stage
        err_softmax = train_epoch_x_vec(train_loader=train_loader,
                                        model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        cuda=cuda,
                                        epoch=epoch)

        # Validation Stage
        accu, cm = val_epoch_x_vec(
            val_loader=val_loader,
            model=model,
            cuda=cuda
        )
        print(', validation accuracy of offline x-vector training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs))

        log_save_name_1 = 'x_vector_training' + exp_time + '.txt'
        with open(os.path.join('results/output_training_log', log_save_name_1), 'a') as f:
            train_output = '\r epoch: [%d / %d], err_softmax: %f' % (epoch + 1, n_epochs, err_softmax)
            f.write(train_output)
            test_output = ', validation accuracy of offline training: %.2f %% for %d / %d' % (accu * 100, epoch + 1, n_epochs)
            f.write(test_output)

    model_name = 'x_vector_model_' + exp_time + '_' + str(format(accu, '.2f')) + '.pkl'
    model_save_path = os.path.join('results/output_model', model_name)
    torch.save(model, model_save_path)
    print('\nBaseline Model saved as:', model_save_path)

    plot_confusion_matrix(cm=cm, savename=model_name + '-confusion_matrix.png', classes=[str(i) for i in range(0,43)])


# Train an epoch on the x-vector
def train_epoch_x_vec(train_loader, model, loss_fn, optimizer, cuda, epoch):
    src_loader = train_loader[0]
    tgt_loader = train_loader[1]

    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)

    len_dataloader = max(len(src_loader), len(tgt_loader)) - 1

    for i in range(len_dataloader):
        model.zero_grad()

        src_data, src_label = src_iter.next()
        src_data = src_data.squeeze()  # 删除channel dimension
        tgt_data, tgt_label = tgt_iter.next()
        tgt_data = tgt_data.squeeze()  # 删除channel dimension

        if cuda:
            src_data = src_data.cuda()
            src_label = src_label.cuda()
            tgt_data = tgt_data.cuda()
            tgt_label = tgt_label.cuda()

        # train_data = src_data
        # train_label = src_label

        train_data = torch.cat((src_data, tgt_data), dim=0)
        train_label = torch.cat((src_label, tgt_label), dim=0)

        train_label = train_label - 1

        # 前向传播
        _, _, pred = model(input_data=train_data)

        # Softmax Loss
        err_softmax = loss_fn(pred, train_label)

        # 反向传播
        err_softmax.backward()
        optimizer.step()

        # Console output
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_softmax: %f' %
                         (epoch,
                          i + 1,
                          len_dataloader,
                          err_softmax.data.cpu().numpy()))
        sys.stdout.flush()

    return err_softmax


# Validate the x-vector
def val_epoch_x_vec(val_loader, model, cuda):
    n_total = 0
    n_correct = 0

    src_val_loader = val_loader[0]
    tgt_val_loader = val_loader[1]

    src_val_iter = iter(src_val_loader)
    tgt_val_iter = iter(tgt_val_loader)

    with torch.no_grad():
        model.eval()

        src_val_data, src_val_label = src_val_iter.next()
        src_val_data = src_val_data.squeeze()  # 删除channel dimension
        src_val_label = np.array(src_val_label.cpu())

        tgt_val_data, tgt_val_label = tgt_val_iter.next()
        tgt_val_data = tgt_val_data.squeeze()  # 删除channel dimension
        tgt_val_label = np.array(tgt_val_label.cpu())

        if cuda:
            src_val_data = src_val_data.cuda()
            tgt_val_data = tgt_val_data.cuda()

        # val_data = src_val_data
        # val_label = src_val_label

        val_data = torch.cat((src_val_data, tgt_val_data), dim=0)
        val_label = np.concatenate((src_val_label, tgt_val_label), axis=0)

        val_label = val_label - 1

        # 前向传播
        _, _, pred_logit = model(input_data=val_data)

        # Inference
        pred_label = np.argmax(pred_logit.detach().cpu().numpy(), axis=1)

        # Statistic
        n_correct += sum(val_label == pred_label)
        n_total += len(val_label)
        accu = n_correct / n_total

        # Confusion Matrix
        confusion_mat = confusion_matrix(val_label, pred_label)

        return accu, confusion_mat
