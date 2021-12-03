import torch
import torch.nn as nn

import numpy as np
import sys
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix


def netFT_fit(train_loader, test_loader, support_label_set, model, loss_fn, optimizer, scheduler, n_epochs, cuda):
    exp_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    log_save_name_1 = 'xvec_dann_orig_FineTuning' + exp_time + '.txt'
    print('\n--------- Start fine-tuning at:' + exp_time + '---------')

    for epoch in range(n_epochs):
        scheduler.step()
        # Train stage
        netFT_train_epoch(
            train_loader=train_loader, model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            cuda=cuda,
            support_label_set=support_label_set)
        # Test stage
        test_accu, _ = netFT_test_epoch(
            test_loader=test_loader,
            model=model,
            cuda=cuda,
            support_label_set=support_label_set)
        print(', Test accuracy: %.2f %% for %d / %d' % (test_accu * 100, epoch + 1, n_epochs))


def netFT_train_epoch(train_loader, model, loss_fn, optimizer, cuda, epoch, support_label_set):
    model.train()
    len_dataloader = len(train_loader)

    for index, item in enumerate(train_loader):
        spt_x, spt_y = item
        spt_x = spt_x.squeeze(1)  # 删除channel dimension

        # # 调整label的值，以进行NLLLoss的计算
        # for i, label_ in enumerate(label):
        #     label[i] = torch.tensor(support_label_set.index(label[i]))

        if cuda:
            model.cuda()
            spt_x = spt_x.cuda()
            spt_y = spt_y.cuda()

        model.zero_grad()
        pred = model(spt_x)
        # _, _, pred = model(sound)
        err_label = loss_fn(pred, spt_y)
        err_label.backward()
        optimizer.step()

        # 打印信息
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_label: %f' %
                         (epoch,
                          index + 1,
                          len_dataloader,
                          err_label.data.cpu().numpy()))
        sys.stdout.flush()
        # print('\r epoch: %d, [iter: %d / all %d], err_label: %f' %
        #                  (epoch,
        #                   index + 1,
        #                   len_dataloader,
        #                   err_label.data.cpu().numpy()))
    return err_label


def netFT_test_epoch(test_loader, model, cuda, support_label_set):
    n_total = 0
    n_correct = 0

    with torch.no_grad():
        model.eval()

        test_iter = iter(test_loader)
        tgt_x, tgt_y = test_iter.next()

        # 调整label的值，以进行NLLLoss的计算
        # for i, label_ in enumerate(test_label):
        #     test_label[i] = torch.tensor(support_label_set.index(test_label[i]))

        tgt_x = tgt_x.squeeze()  # 删除channel dimension
        tgt_y = np.array(tgt_y.cpu())

        if cuda:
            tgt_x = tgt_x.cuda()

        pred = model(tgt_x)

        pred_label = np.argmax(np.array(pred.cpu()), axis=1)

        n_correct += sum(tgt_y == pred_label)
        n_total += len(tgt_y)
        accu = n_correct / n_total

        confusion_mat = confusion_matrix(tgt_y, pred_label)

    return accu, confusion_mat
